"""
Phase D — Time integration of the DO-reduced 2D Navier–Stokes system

This solver advances in time:
  • the mean fields  Ū(x,y,t), V̄(x,y,t)  (PDE, explicit Euler in spectral space).
  • the S spatial modes  {Φ_i(x,y,t)}_{i=1..S}, each a 2-component field (PDE, explicit Euler in spectral).
  • the stochastic coefficients  Y(t) ∈ R^{MC×S}  (ODE system, low-storage RK4).

Key ingredients
---------------
• Spectral advection terms are computed with 3/2 de-aliasing (aa_product).
• The Leray (Helmholtz) projector removes pressure (divergence-free constraint).
• Inner products use the spectral Parseval form (inner_product_spectral).
• Each step: re-orthonormalize modes in spectral space (orthonormalize_spectral),
  and optionally diagonalize C_YY (diag_cov_rotate).
"""

from __future__ import annotations
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np

from .grids import build_grids, projection_multipliers
from .spectral import aa_product, project_div_free, enforce_hermitian_symmetry
from .coloured_noise_forcing import make_forcing
from .params import Params
from .stats import (
    inner_product_spectral,
    second_moment,
    third_moment,
    orthonormalize_spectral_qr,
    diag_cov_rotate,
)


@dataclass
class Result:
    """Container for solver output."""
    T_save: np.ndarray        # (Nsav,)
    CYY_save: np.ndarray      # (1+S, Nsav)   [energy in mean; diag(CYY)]
    U_save: np.ndarray        # (Ny, Nx, Nsav)
    V_save: np.ndarray        # (Ny, Nx, Nsav)
    Ui_save: np.ndarray       # (Ny, Nx, S, Nsav)
    Vi_save: np.ndarray       # (Ny, Nx, S, Nsav)
    YY_save: np.ndarray       # (MC, S, Nsav)
    half_k_pairs: np.ndarray  # (M, 2) where M = len(half)
    ou_amplitudes: np.ndarray
    T_plot: np.ndarray        # (Nplot,)
    CYY_plot: np.ndarray      # (1+S, Nplot)
    W_save: np.ndarray        # (Ny, Nx, Nsav)
    Wi_save: np.ndarray       # (Ny, Nx, S, Nsav)


def _rk4_low_storage_coeffs():
    """Kennedy–Carpenter low-storage RK4 coefficients/"""
    rk4a = np.array([
        0.0,
        -567301805773.0/1357537059087.0,
        -2404267990393.0/2016746695238.0,
        -3550918686646.0/2091501179385.0,
        -1275806237668.0/842570457699.0,
    ], dtype=float)
    rk4b = np.array([
        1432997174477.0/9575080441755.0,
        5161836677717.0/13612068292357.0,
        1720146321549.0/2090206949498.0,
        3134564353537.0/4481467310338.0,
        2277821191437.0/14882151754819.0,
    ], dtype=float)
    return rk4a, rk4b


def solve_do(
    Uh: np.ndarray,
    Vh: np.ndarray,
    Uih: np.ndarray,
    Vih: np.ndarray,
    YY: np.ndarray,
    Gxh_det: np.ndarray,
    Gyh_det: np.ndarray,
    P: Params,
    plotter: Optional[object] = None
) -> Result:
    """
    Advance the DO-reduced system in time.

    Parameters
    ----------
    Uh, Vh : (Ny, Nx) float
        Mean velocity components at t=0 in spectral space.
    Uih, Vih : (Ny, Nx, S) float
        Initial DO modes in spectral space.
    YY : (MC, S) float
        Initial stochastic coefficients (demeaned).
    Gxh_det, Gyh_det : (Ny, Nx) complex
        Spectral stochastic_forcing (FFT2 of Gx, Gy).
    P : Params
        Parameters (Lx,Ly; Nx,Ny; Tf,dt; S,MC; Re; PlotInterval, SaveInterval; DiagModes).

    Returns
    -------
    Result
        A dataclass carrying saved fields/times and energy diagnostics.
    """
    Ny, Nx = P.Ny, P.Nx
    S = P.S
    MC = P.MC

    # --- Grids / spectral operators ---
    x, y, kx, ky, k2, dx, dy = build_grids(P)
    p1, p2, p3 = projection_multipliers(kx, ky)
    ikx, iky = 1j * kx, 1j * ky

    # --- Allocate workspace for antialiasing ---
    hNy, hNx = Ny // 2, Nx // 2
    My, Mx = Ny + hNy, Nx + hNx
    aa_ws = (
        np.zeros((My, Mx), dtype=np.complex128),  # Uh_pad
        np.zeros((My, Mx), dtype=np.complex128),  # Vh_pad
        np.zeros((Ny, Nx), dtype=np.complex128),  # out
    )

    # --- Initialise stochastic forcing + relevant parameters ---
    stochastic_forcing = make_forcing(P)
    half = (np.array(stochastic_forcing[0].half)
            if isinstance(stochastic_forcing, list) and stochastic_forcing else None)
    ou_amplitudes = []
    if isinstance(stochastic_forcing, list) and stochastic_forcing:
        K_T = len(stochastic_forcing[0].half)
        ou_amplitudes.append(np.zeros((MC, K_T), dtype=float))

    # --- Storage for nonlinear terms ---
    PFihND = np.zeros_like(Uih)                              # (Ny, Nx, S)
    PGihND = np.zeros_like(Vih)
    PFijh = np.zeros((Ny, Nx, S, S), dtype=np.complex128)    # (Ny, Nx, S, S)
    PGijh = np.zeros_like(PFijh)

    # --- Time bookkeeping and diagnostics buffers ---
    Nt = int(np.ceil(P.Tf / P.dt))
    nplot = 1 + Nt // max(1, P.PlotInterval)
    nsave = 1 + Nt // max(1, P.SaveInterval)

    T_plot = np.zeros(nplot, dtype=float)
    CYY_plot = np.zeros((1 + S, nplot), dtype=float)

    T_save = np.zeros(nsave, dtype=float)
    CYY_save = np.zeros((1 + S, nsave), dtype=float)
    U_save = np.zeros((Ny, Nx, nsave), dtype=float)
    V_save = np.zeros((Ny, Nx, nsave), dtype=float)
    Ui_save = np.zeros((Ny, Nx, S, nsave), dtype=float)
    Vi_save = np.zeros((Ny, Nx, S, nsave), dtype=float)
    YY_save = np.zeros((MC, S, nsave), dtype=float)
    W_save = np.zeros((Ny, Nx, nsave), dtype=float)
    Wi_save = np.zeros((Ny, Nx, S, nsave), dtype=float)

    # --- Initial diagnostics ---
    CYY = second_moment(YY)  # (S, S)
    energy_mean = 0.5 * inner_product_spectral(Uh, Vh, Uh, Vh, P)
    CYY_plot[:, 0] = np.concatenate(([energy_mean], np.diag(CYY)))
    CYY_save[:, 0] = CYY_plot[:, 0]
    T_plot[0] = 0.0
    T_save[0] = 0.0
    U_save[..., 0] = np.fft.ifft2(Uh).real
    V_save[..., 0] = np.fft.ifft2(Vh).real
    Ui_save[..., 0] = np.fft.ifft2(Uih).real
    Vi_save[..., 0] = np.fft.ifft2(Vih).real
    YY_save[..., 0] = YY
    W_save[..., 0] = np.fft.ifft2(ikx * Vh - iky * Uh).real
    Wi_save[..., 0] = np.fft.ifft2(ikx[..., None] * Vih - iky[..., None] * Uih, axes=(0, 1)).real
    if plotter is not None:
        plotter.update(
            0.0, Uh, Vh, Uih, Vih, YY,
            energy_mean=float(energy_mean),
            energies_modes=np.diag(CYY).copy()
        )

    # --- RK coefficients for Y update ---
    rk4a, rk4b = _rk4_low_storage_coeffs()

    # --- Time loop ---
    iplt = 1
    isav = 1
    import time
    t_start = time.perf_counter()
    t_prev = t_start

    for nt in range(1, Nt + 1):
        t = nt * P.dt
        K = 100
        if nt % K == 0:
            t_now = time.perf_counter()
            avg_dt = (t_now-t_prev)/ K
            total_avg = (t_now - t_start) / nt
            print(f"{nt}/{Nt} | avg dt (last {K}): {avg_dt:.3f}s | avg dt (total): {total_avg:.3f}s", flush=True)
            t_prev = t_now

        # Snapshot of current spectral variables
        Uh0 = Uh.copy()
        Vh0 = Vh.copy()
        Uih0 = Uih.copy()
        Vih0 = Vih.copy()

        # Compute second & third moments
        CYY = second_moment(YY)              # (S,S)
        MYYY = third_moment(YY)              # (S,S,S)

        # -------------------------------
        # 0) Build stochastic forcing if present
        # -------------------------------
        a_vec = None
        if isinstance(stochastic_forcing, list) and stochastic_forcing:
            fxh_s = np.empty((MC, Ny, Nx), dtype=np.complex128) # OU amplitudes for this timestep for each realisation
            fyh_s = np.empty_like(fxh_s)
            amps_list = []
            for s, F in enumerate(stochastic_forcing):
                fxh_s[s], fyh_s[s] = F.step_hat()
                amps_list.append(P.ou_amp * np.array([ou_j.x + ou_j.mu for ou_j in F.ou], dtype=float))
            a_vec = np.stack(amps_list, axis=0)

        # -------------------------------
        # 1) Build nonlinear terms (AA + projection) for mean and modes
        # -------------------------------
        # Mean self-interaction (de-aliased)
        AA11 = aa_product(Uh0, Uh0, aa_ws).copy()                # \widehat{uu}
        AA12 = aa_product(Uh0, Vh0, aa_ws).copy()                # \widehat{uv}
        AA22 = aa_product(Vh0, Vh0, aa_ws).copy()                # \widehat{vv}

        # Build RHS terms independent of stochastic coefficients
        F0hND_det = -(ikx * AA11 + iky * AA12) + Gxh_det
        G0hND_det = -(ikx * AA12 + iky * AA22) + Gyh_det
        PF0hND_det, PG0hND_det = project_div_free(F0hND_det, G0hND_det, p1, p2, p3)

        # Mean–mode interactions (linear in modes)
        for m in range(S):
            Umh = Uih0[..., m]
            Vmh = Vih0[..., m]

            # Mixed AA products (de-aliased)
            A1 = aa_product(Uh0, Umh, aa_ws).copy()                                             # u ⋆ u_m
            A2 = aa_product(Uh0, Vmh, aa_ws).copy() + aa_product(Vh0, Umh, aa_ws).copy()               # u ⋆ v_m + v ⋆ u_m
            A3 = aa_product(Vh0, Vmh, aa_ws).copy()                                             # v ⋆ v_m

            # (ND: non-diffusive, i.e. convective + forcing)
            FmhND = -(2 * ikx * A1 + 1 * iky * A2)
            GmhND = -(1 * ikx * A2 + 2 * iky * A3)

            PFihND[..., m], PGihND[..., m] = project_div_free(FmhND, GmhND, p1, p2, p3)

            # Mode–mode interactions (quadratic in modes)
            for n in range(S):
                Unh = Uih0[..., n]
                Vnh = Vih0[..., n]

                B1 = aa_product(Umh, Unh, aa_ws).copy()
                B2 = aa_product(Umh, Vnh, aa_ws).copy()
                B3 = aa_product(Vmh, Vnh, aa_ws).copy()

                Fmnh = -(ikx * B1 + iky * B2)
                Gmnh = -(ikx * B2 + iky * B3)

                PFijh[..., m, n], PGijh[..., m, n] = project_div_free(Fmnh, Gmnh, p1, p2, p3)

        # Add viscous terms for modes (mean handled explicitly below)
        PFih = - (k2[..., None] / P.Re) * Uih0 + PFihND
        PGih = - (k2[..., None] / P.Re) * Vih0 + PGihND

        # -------------------------------
        # 2) Build Galerkin inner-product tensors A_im and A_imn
        # -------------------------------
        U = Uih0.reshape(Ny*Nx, S)
        V = Vih0.reshape(Ny*Nx, S)
        FU = (-(k2[..., None] / P.Re) * Uih0 + PFihND).reshape(Ny*Nx, S)
        FV = (-(k2[..., None] / P.Re) * Vih0 + PGihND).reshape(Ny*Nx, S)
        c = (P.Lx * P.Ly) / (P.Nx * P.Ny)**2
        Aim = c * (U.conj().T @ FU + V.conj().T @ FV).real
        PFijh_rs = PFijh.reshape(Ny*Nx, S*S)
        PGijh_rs = PGijh.reshape(Ny*Nx, S*S)
        G_quad = c * (U.conj().T @ PFijh_rs + V.conj().T @ PGijh_rs)
        Aimn = G_quad.reshape(S,S,S).real

        # -------------------------------
        # 3) Advance mean (explicit Euler, spectral)
        #     d/dt Uh = -k2/Re * Uh + PF0hND + sum_{m,n} PFijh_{m,n} CYY_{mn}
        # -------------------------------
        forcing_covF = np.tensordot(PFijh, CYY, axes=([2, 3], [0, 1]))  # (Ny, Nx)
        forcing_covG = np.tensordot(PGijh, CYY, axes=([2, 3], [0, 1]))  # (Ny, Nx)

        Uh = Uh0 + P.dt * (-(k2 * Uh0) / P.Re + PF0hND_det + forcing_covF)
        Vh = Vh0 + P.dt * (-(k2 * Vh0) / P.Re + PG0hND_det + forcing_covG)

        # -------------------------------
        # 4) Advance modes (explicit Euler)
        #     H_i = PFih_i + sum_{m,n} PFijh_{m,n} (MYYY_{mn·}) CYY^{-1}
        #     then orthogonalize against old basis
        # -------------------------------
        # Build Hih, Kih via third moment contraction and CYY^{-1}
        # CYY_inv = np.linalg.inv(CYY)
        # M3_2d = MYYY.reshape(S * S, S) @ CYY_inv
        M3_2d = np.linalg.solve(CYY.T, MYYY.reshape(S * S, S).T).T

        addF = PFijh_rs @ M3_2d
        addG = PGijh_rs @ M3_2d
        addF = addF.reshape(Ny, Nx, S)  # M_{klj} \widehat{PF_{kl}} C_{ij}^{-1}
        addG = addG.reshape(Ny, Nx, S)  # M_{klj} \widehat{PG_{kl}} C_{ij}^{-1}

        addStochF = np.zeros((Ny, Nx, S), dtype=np.complex128)
        addStochG = np.zeros_like(addStochF)

        if isinstance(stochastic_forcing, list) and stochastic_forcing:
            EPfYx = (fxh_s[:, :, :, None] * YY[:, None, None, :]).mean(axis=0)
            EPfYy = (fyh_s[:, :, :, None] * YY[:, None, None, :]).mean(axis=0)

            addStochF = np.linalg.solve(CYY.T, EPfYx.reshape(-1, S).T).T.reshape(Ny, Nx, S)
            addStochG = np.linalg.solve(CYY.T, EPfYy.reshape(-1, S).T).T.reshape(Ny, Nx, S)

        Hih = PFih + addF + addStochF
        Kih = PGih + addG + addStochG

        H = Hih.reshape(Ny*Nx, S)
        K = Kih.reshape(Ny*Nx, S)
        Proj = c * (U.conj().T @ H + V.conj().T @ K).real
        Uih = Uih0 + P.dt * (Hih - np.tensordot(Uih0, Proj, axes=(2,0)))
        Vih = Vih0 + P.dt * (Kih - np.tensordot(Vih0, Proj, axes=(2,0)))

        # -------------------------------
        # 5) Advance Y with low-storage RK4
        #     dY_i/dt = sum_m Y_m A_im + sum_{m,n} (Y_m Y_n - CYY_{mn}) A_imn
        # -------------------------------
        # Forcing projection for dY_i/dt: b[s,i] = < (fxh_s, fyh_s), (Uih0[...,i], Vih0[...,i]) >
        if isinstance(stochastic_forcing, list) and stochastic_forcing:
            Fx = fxh_s.reshape(MC, Ny*Nx)
            Fy = fyh_s.reshape(MC, Ny*Nx)
            b = c * (Fx @ U.conj() + Fy @ V.conj()).real
        else:
            b = 0.0

        rk_res = np.zeros_like(YY)     # (MC, S)
        CYY_stage = CYY
        for irk in range(5):
            RHS = np.zeros_like(YY)
            # Stochastic part
            RHS += b
            # Linear part: Y @ A_im^T
            RHS += YY @ Aim.T
            # Quadratic part (vectorised): for each i, sum_{m,n} (Y_m ⊙ Y_n - CYY_mn) * A_imn
            term1 = np.einsum('pm,pn,imn->pi', YY, YY, Aimn, optimize=True)
            term2 = np.einsum('mn,imn->i', CYY_stage, Aimn, optimize=True)
            RHS += term1 - term2[None, :]
            # Low-storage RK4 update
            rk_res = rk_res * _rk4a_const(irk) + P.dt * RHS
            YY = YY + _rk4b_const(irk) * rk_res

            CYY_stage = (YY.T @ YY) / MC

        # -------------------------------
        # 6) Enforce constraints & DO conditions
        # -------------------------------
        # --- Enforce Hermitian symmetry (real) and divergence-free fields ---
        Uh = enforce_hermitian_symmetry(Uh); Vh = enforce_hermitian_symmetry(Vh)
        Uh, Vh = project_div_free(Uh, Vh, p1, p2, p3)
        for s in range(S):
            Uih[..., s] = enforce_hermitian_symmetry(Uih[..., s])
            Vih[..., s] = enforce_hermitian_symmetry(Vih[..., s])
            Uih[..., s], Vih[..., s] = project_div_free(Uih[..., s], Vih[..., s], p1, p2, p3)

        # --- Enforce mode orthonormality ---
        YY, Uih, Vih= orthonormalize_spectral_qr(YY, Uih, Vih, P)

        # --- Recentre coefficients (compensating mean shift) ---
        mu = YY.mean(axis=0)
        Uh += np.tensordot(Uih, mu, axes=(2,0))
        Vh += np.tensordot(Vih, mu, axes=(2,0))
        YY -= mu

        # --- Diagonalise ---
        YY, Uih, Vih = diag_cov_rotate(YY, Uih, Vih, P) # rotate by eigvecs of CYY

        # -------------------------------
        # 7) Plot/save bookkeeping
        # -------------------------------
        if (nt % max(1, P.PlotInterval)) == 0:
            ip = iplt
            T_plot[ip] = t
            CYY = second_moment(YY)
            energy_mean = inner_product_spectral(Uh, Vh, Uh, Vh, P)
            CYY_plot[:, ip] = np.concatenate(([energy_mean], np.diag(CYY)))
            if plotter is not None:
                plotter.update(
                    t, Uh, Vh, Uih, Vih, YY,
                    energy_mean=float(energy_mean),
                    energies_modes=np.diag(CYY).copy()
                )
            iplt += 1

        if (nt % max(1, P.SaveInterval)) == 0:
            isv = isav
            T_save[isv] = t
            CYY = second_moment(YY)
            energy_mean = 0.5 * inner_product_spectral(Uh, Vh, Uh, Vh, P)
            CYY_save[:, isv] = np.concatenate(([energy_mean], np.diag(CYY)))

            U_save[..., isv] = np.fft.ifft2(Uh).real
            V_save[..., isv] = np.fft.ifft2(Vh).real
            for m in range(S):
                Ui_save[..., m, isv] = np.fft.ifft2(Uih[..., m]).real
                Vi_save[..., m, isv] = np.fft.ifft2(Vih[..., m]).real
            YY_save[..., isv] = YY
            W_save[..., isv] = np.fft.ifft2(ikx * Vh - iky * Uh).real
            Wi_save[..., isv] = np.fft.ifft2(ikx[..., None] * Vih - iky[..., None] * Uih, axes=(0, 1)).real
            ou_amplitudes.append(a_vec)
            isav += 1

        if nt == 10:
            break

        # # Blow-up guard
        # if np.max(np.abs(Uh)) > 1e10 or np.max(np.abs(Vh)) > 1e10:
        #     # fill remaining plot/save arrays up to current indices and break
        #     print(t)
        #     break

    # Trim arrays to the last written indices (in case Tf not multiple of intervals)
    T_plot = T_plot[:iplt]
    CYY_plot = CYY_plot[:, :iplt]
    T_save = T_save[:isav]
    CYY_save = CYY_save[:, :isav]
    U_save = U_save[..., :isav]
    V_save = V_save[..., :isav]
    Ui_save = Ui_save[..., :isav]
    Vi_save = Vi_save[..., :isav]
    YY_save = YY_save[..., :isav]
    W_save = W_save[..., :isav]
    Wi_save = Wi_save[..., :isav]
    if isinstance(stochastic_forcing, list) and stochastic_forcing:
        ou_amplitudes = np.stack(ou_amplitudes, axis=0)

    return Result(
        T_save=T_save,
        CYY_save=CYY_save,
        U_save=U_save,
        V_save=V_save,
        Ui_save=Ui_save,
        Vi_save=Vi_save,
        YY_save=YY_save,
        half_k_pairs=half,
        ou_amplitudes=ou_amplitudes,
        T_plot=T_plot,
        CYY_plot=CYY_plot,
        W_save=W_save,
        Wi_save=Wi_save
    )


# --- Helpers for RK constants to avoid re-alloc in the loop ---
_RK4A, _RK4B = _rk4_low_storage_coeffs()

def _rk4a_const(i: int) -> float:
    return float(_RK4A[i])

def _rk4b_const(i: int) -> float:
    return float(_RK4B[i])
