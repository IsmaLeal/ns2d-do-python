from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

from .params import Params
from .grids import build_grids, projection_multipliers
from .spectral import project_div_free
from .stats import diag_cov_rotate, orthonormalize_spectral_qr


def _mollifier(Ny: int, Nx: int) -> np.ndarray:
    """
    Edge-taper "mollifier" that damps the vortex near the domain edges to improve periodicity.

    Builds a (Ny+1, Nx+1) grid with zeros at boundaries, then performs 4 smoothing
    passes using an 8-neighbor average multiplied by the center value, finally
    drops the last row/column to return a (Ny, Nx) array.

    Parameters
    ----------
    Ny, Nx : int
        Physical grid dimensions.

    Returns
    -------
    mol : (Ny, Nx) float
        Smooth mask in [0, 1] that damps the field near the domain edges.
    """
    Ny1, Nx1 = Ny + 1, Nx + 1
    mol = np.ones((Ny1, Nx1), dtype=float)
    mol[0, :] = 0.0
    mol[-1, :] = 0.0
    mol[:, 0] = 0.0
    mol[:, -1] = 0.0
    for _ in range(4):
        mol_old = mol.copy()
        # Interior: center * average of 8 neighbors
        mol[1:-1, 1:-1] = mol_old[1:-1, 1:-1] * (
            mol_old[0:-2, 1:-1] + mol_old[0:-2, 0:-2] + mol_old[0:-2, 2:] +
            mol_old[1:-1, 0:-2] + mol_old[1:-1, 2:] +
            mol_old[2:, 0:-2] + mol_old[2:, 1:-1] + mol_old[2:, 2:]
        ) / 8.0
    return mol[:-1, :-1]  # drop last row/col


def _forcing_arrays(P: Params, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build physical-space forcing arrays (Gx, Gy) on the (Ny, Nx) grid.

    Parameters
    ----------
    P : Params
        Parameter set (Forcing ∈ {"none","kick","kolmogorov"}).
    x, y : (Nx,), (Ny,) float
        1D coordinate arrays (periodic).

    Returns
    -------
    Gx, Gy : (Ny, Nx) float
        Forcing components in physical space.
    """
    xx, yy = np.meshgrid(x, y, indexing="xy")
    forcing = P.forcing.lower()
    amp = P.f_amp
    if forcing == "none":
        Gx = np.zeros_like(xx)
        Gy = np.zeros_like(xx)
    elif forcing == "kick":
        Gx = amp * np.exp(-4.0 * (xx - P.Lx / 2.0) ** 2 - 4.0 * (yy - P.Ly / 2.0) ** 2) * (
            2.0 + np.tanh(yy - P.Ly / 2.0)
        )
        Gy = np.zeros_like(xx)
    elif forcing == "kolmogorov":
        Gx = amp * np.sin(4 * yy)
        Gy = np.zeros_like(xx)
    else:
        raise ValueError(f"Unknown Forcing '{P.forcing}'")
    return Gx, Gy


def initial_conditions(
        P: Params,
        MR: Optional[int] = None,
        seed: Optional[int] = None,
        plotIC: Optional[int] = 0,
        eps_IC_noise: Optional[float] = 0.09,
        smallvar: Optional[bool] = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct initial mean fields, DO modes, coefficients, and spectral forcing.

    This is a faithful translation of MATLAB `SetupScript.m`:
    - Build MR realizations of a Lamb–Oseen-like swirl.
    - Project each realization to divergence-free using the spectral Leray projector.
    - Compute mean (U,V) and anomalies (M,N).
    - Compute top S left singular vectors of X=[M;N] via economy SVD → modes (Ui,Vi).
    - Project samples1 onto modes → Y, then de-mean and keep first MC rows.

    Parameters
    ----------
    P : Params
        Parameters (domain, grid, S, MC, Re, AdvecAngle, Forcing, Seed, ...).
    MR : int, optional
        Number of realizations for the ensemble.
        Default: max(1000, P.MC), matching the MATLAB code.
    seed : int, optional
        RNG seed override (falls back to P.Seed).

    Returns
    -------
    U, V : (Ny, Nx) float
        Mean velocity components in physical space.
    Ui, Vi : (Ny, Nx, S) float
        Initial DO modes (physical space).
    YY : (MC, S) float
        Initial stochastic coefficients (demeaned).
    Gxh, Gyh : (Ny, Nx) complex
        Spectral forcing (FFT2 of Gx and Gy).
    """
    # --- Grids and projector multipliers ---
    x, y, kx, ky, k2, dx, dy = build_grids(P)
    p1, p2, p3 = projection_multipliers(kx, ky)

    # --- Forcing and their FFTs ---
    Gx, Gy = _forcing_arrays(P, x, y)
    Gxh = np.fft.fft2(Gx)
    Gyh = np.fft.fft2(Gy)

    # --- RNG and mollifier (edge taper) ---
    rng = np.random.default_rng(P.Seed if seed is None else seed)
    mol = _mollifier(P.Ny, P.Nx)

    # --- Ensemble size (MR) ---
    MR_eff = int(MR) if MR is not None else max(1000, int(P.MC))

    # --- Build realizations and project each to div-free ---
    Ny, Nx = P.Ny, P.Nx
    xx, yy = np.meshgrid(x, y, indexing="xy")
    M = np.zeros((Ny, Nx, MR_eff), dtype=float)  # u-samples1
    N = np.zeros((Ny, Nx, MR_eff), dtype=float)  # v-samples1

    if eps_IC_noise == 0:
        for iR in range(MR_eff):
            # Parameters for the swirl
            gamma = 10.0
            if not smallvar:
                rc = abs(0.2 + rng.normal(0.0, 0.05))
            else:
                rc = abs(0.2 + rng.normal(0.0, 1e-4))
            x0 = P.Lx / 2.0
            y0 = P.Ly / 2.0

            # Polar radius/angle relative to center (avoid rr=0 division)
            rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
            rr_safe = np.where(rr == 0.0, 1.0, rr)
            ttheta = np.arctan2(yy - y0, xx - x0)

            # Lamb–Oseen-like tangential speed; set center value explicitly to 0
            vtheta = gamma / (2.0 * np.pi * rr_safe) * (1.0 - np.exp(-(rr ** 2) / (rc ** 2)))
            vtheta[rr == 0.0] = 0.0

            U_phys = -vtheta * np.sin(ttheta) * mol
            V_phys = vtheta * np.cos(ttheta) * mol

            # Spectral projection to divergence-free
            Uh = np.fft.fft2(U_phys)
            Vh = np.fft.fft2(V_phys)
            PUh, PVh = project_div_free(Uh, Vh, p1, p2, p3)

            # Back to physical space (real fields)
            M[:, :, iR] = np.fft.ifft2(PUh).real
            N[:, :, iR] = np.fft.ifft2(PVh).real

        # --- Means and anomalies ---
        U_sample = M.mean(axis=2)
        V_sample = N.mean(axis=2)
        U = U_sample.copy()
        V = V_sample.copy()
        M -= U_sample[:, :, None]
        N -= V_sample[:, :, None]

        # --- Stack anomalies and compute SVD for modes ---
        Npts = Ny * Nx
        X = np.empty((2 * Npts, MR_eff), dtype=float)
        X[:Npts, :] = M.reshape(Npts, MR_eff)
        X[Npts:, :] = N.reshape(Npts, MR_eff)

        U_svd, svals, Vt = np.linalg.svd(X, full_matrices=False)
        U_svd = U_svd[:, :P.S]
        Ui = U_svd[:Npts, :].reshape(Ny, Nx, P.S)
        Vi = U_svd[Npts:, :].reshape(Ny, Nx, P.S)
        YY_full = X.T @ U_svd

        YY = YY_full[:P.MC, :]
        mu = YY.mean(axis=0)
        U += np.tensordot(Ui, mu, axes=(2, 0))
        V += np.tensordot(Vi, mu, axes=(2, 0))
        YY -= mu

        Uh = np.fft.fft2(U); Vh = np.fft.fft2(V)
        Uih = np.fft.fft2(Ui); Vih = np.fft.fft2(Vi)
        YY, Uih, Vih = orthonormalize_spectral_qr(YY, Uih, Vih, P)
        YY, Uih, Vih = diag_cov_rotate(YY, Uih, Vih, P)

    if eps_IC_noise != 0:
        # excite |k| <= k_max
        k_max = 8
        # deterministic initial vortex
        gamma = 10.0
        rc = 0.2
        x0 = P.Lx / 2.0
        y0 = P.Ly / 2.0
        mol = _mollifier(P.Ny, P.Nx)

        rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
        rr_safe = np.where(rr == 0.0, 1.0, rr)
        ttheta = np.arctan2(yy - y0, xx - x0)
        vtheta = gamma / (2.0 * np.pi * rr_safe) * (1.0 - np.exp(-(rr ** 2) / (rc ** 2)))
        vtheta[rr == 0.0] = 0.0

        Uh_base = np.fft.fft2(-vtheta * np.sin(ttheta) * mol)
        Vh_base = np.fft.fft2(vtheta * np.cos(ttheta) * mol)
        PUh_base, PVh_base = project_div_free(Uh_base, Vh_base, p1, p2, p3)
        U_base = np.fft.ifft2(PUh_base).real
        V_base = np.fft.ifft2(PVh_base).real

        # --- build MR_eff realisations ---
        M = np.zeros((Ny, Nx, MR_eff), dtype=float)
        N = np.zeros((Ny, Nx, MR_eff), dtype=float)

        K = np.sqrt(k2)
        mask = (K > 0.0) & (K <= k_max)

        for iR in range(MR_eff):
            # --- real white noise -> FFT -> band-limit to |k|<=k_max ---
            # generate white noise fields
            uW = rng.standard_normal((Ny, Nx))
            vW = rng.standard_normal((Ny, Nx))
            # FFT and band-limit
            UhN = np.fft.fft2(uW) * mask
            VhN = np.fft.fft2(vW) * mask
            # div-free perturbation
            PUhN, PVhN = project_div_free(UhN, VhN, p1, p2, p3)
            # iFFT back to physical space
            du = np.fft.ifft2(PUhN).real
            dv = np.fft.ifft2(PVhN).real
            # RMS control
            rms = np.sqrt(np.mean(du**2 + dv**2))
            if rms > 0:
                du /= rms
                dv /= rms
            # add perturbation to base IC
            U_phys = U_base + eps_IC_noise * du
            V_phys = V_base + eps_IC_noise * dv

            M[:, :, iR] = U_phys
            N[:, :, iR] = V_phys

        # --- Means and anomalies (same as your main path) ---
        U_sample = M.mean(axis=2)
        V_sample = N.mean(axis=2)
        U = U_sample.copy()
        V = V_sample.copy()
        M -= U_sample[:, :, None]
        N -= V_sample[:, :, None]

        # --- Stack anomalies and compute SVD for modes (same as your main path) ---
        Npts = Ny * Nx
        X = np.empty((2 * Npts, MR_eff), dtype=float)
        X[:Npts, :] = M.reshape(Npts, MR_eff)
        X[Npts:, :] = N.reshape(Npts, MR_eff)

        U_svd, svals, Vt = np.linalg.svd(X, full_matrices=False)
        U_svd = U_svd[:, :P.S]
        Ui = U_svd[:Npts, :].reshape(Ny, Nx, P.S)
        Vi = U_svd[Npts:, :].reshape(Ny, Nx, P.S)
        YY_full = X.T @ U_svd

        YY = YY_full[:P.MC, :]
        mu = YY.mean(axis=0)
        U += np.tensordot(Ui, mu, axes=(2,0))
        V += np.tensordot(Vi, mu, axes=(2,0))
        YY -= mu

        Uh = np.fft.fft2(U); Vh = np.fft.fft2(V)
        Uih = np.fft.fft2(Ui); Vih = np.fft.fft2(Vi)
        YY, Uih, Vih = orthonormalize_spectral_qr(YY, Uih, Vih, P)
        YY, Uih, Vih = diag_cov_rotate(YY, Uih, Vih, P)

    if plotIC:
        # choose realisation
        r = np.random.randint(0, P.MC)

        # plot
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import Normalize
        plt.rcParams.update({"font.size": 17})
        fig = plt.figure(figsize=(15,10), constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig)

        # create each axis
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        for ax in [ax0, ax1]:
            ax.set_aspect("equal")
            ax.set_xlabel("x", fontsize=17, fontfamily="serif")
            ax.set_ylabel("y", fontsize=17, fontfamily="serif")
            ax.set_xlim(0, P.Lx); ax.set_ylim(0, P.Ly)
        ax0.set_title(r"Mean initial velocity field $\mathbf{\overline{u}}(\mathbf{x},t=0)$")
        ax1.set_title(rf"Initial velocity field $\mathbf{{u}}(\mathbf{{x}},t=0)$ of realisation {r}")

        # compute mean and realisation fields
        ubar = U; vbar = V
        ur = ubar + np.einsum("xys,s->xy", Ui, YY[r, :])
        vr = vbar + np.einsum("xys,s->xy", Vi, YY[r, :])

        # plotting parameters
        cmap = plt.get_cmap("turbo")
        extent = (0, P.Lx, 0, P.Ly)
        M_mean = np.max(np.hypot(ubar, vbar))
        M_real = np.max(np.hypot(ur, vr))
        M = max(M_real, M_mean)
        norm = Normalize(vmin=0.0, vmax=M)
        Lscreen = 0.2 * min(P.Lx, P.Ly)
        scale = M / Lscreen
        s = slice(None, None, 4); Xs = xx[s, s]; Ys = yy[s, s]

        speed_mean = np.hypot(ubar, vbar); speed_r = np.hypot(ur, vr)
        ax0.imshow(speed_mean, origin="lower", norm=norm, cmap=cmap, extent=extent, interpolation="bilinear")
        ax1.imshow(speed_r, origin="lower", norm=norm, cmap=cmap, extent=extent, interpolation="bilinear")

        ax0.quiver(Xs, Ys, ubar[s, s], vbar[s, s], color="k", angles="xy", scale_units="xy", scale=scale, units="inches", width=0.02, pivot="mid")
        ax1.quiver(Xs, Ys, ur[s, s], vr[s, s], color="k", angles="xy", scale_units="xy", scale=scale, units="inches", width=0.02, pivot="mid")

        plt.show()
    return Uh, Vh, Uih, Vih, YY, Gxh, Gyh