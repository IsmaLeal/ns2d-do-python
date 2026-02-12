from __future__ import annotations
import numpy as np
from typing import Tuple
from .params import Params


def inner_product_physical(
    U1: np.ndarray,
    V1: np.ndarray,
    U2: np.ndarray,
    V2: np.ndarray,
    P: Params
) -> float:
    """
    Computes the discrete inner product between two fields.
    """
    dx = P.Lx / P.Nx
    dy = P.Ly / P.Ny
    val = (U1.conj() * U2 + V1.conj() * V2).sum().real * dx * dy
    return float(val)


def inner_product_spectral(
    U1h: np.ndarray,
    V1h: np.ndarray,
    U2h: np.ndarray,
    V2h: np.ndarray,
    P: Params
) -> float:
    """
    <(u1,v1),(u2,v2)> via Parseval:
      c * sum(U1h*conj(U2h) + V1h*conj(V2h)),  c = Lx*Ly / (Nx*Ny)^2,
      where Uh is the Fourier transform of vector U.
    """
    c = (P.Lx * P.Ly) / (P.Nx * P.Ny) ** 2
    val = (U1h.conj() * U2h + V1h.conj() * V2h).sum() * c
    return float(np.real(val))


def second_moment(Y: np.ndarray) -> np.ndarray:
    """
    CYY = E[YY^T] with sample mean removed: (Y0^T Y0)/MC.
    Y shape: (MC, S).
    """
    MC = Y.shape[0]
    C = (Y.T @ Y) / MC
    C = 0.5 * (C + C.T)
    return C


def third_moment(Y: np.ndarray) -> np.ndarray:
    """
    M_{mnl} = E[Y_m Y_n Y_l] computed from samples1.
    Y shape: (MC, S).  (Call with the Y used in the solver loop — typically already demeaned.)
    """
    MC = Y.shape[0]
    return np.einsum("im,in,il->mnl", Y, Y, Y, optimize=True) / MC


def orthonormalize_spectral_qr(
    YY: np.ndarray,
    Uih: np.ndarray,
    Vih: np.ndarray,
    P
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    QR-based orthonormalization of spectral modes.

    Inputs
    ------
    Uih, Vih : complex arrays (Ny, Nx, S), where S is the number of modes.
    P        : Params with (Lx, Ly, Nx, Ny)

    Returns
    -------
    Uo, Vo : orthonormal modes (Ny, Nx, S) (in the Euclidean sense; Parseval implies DO-orthonormal)
    R      : (S, S) upper-triangular with Phi = Q R, so update coefficients as Y <- Y @ R.T
    """
    if Uih.shape != Vih.shape or Uih.ndim != 3:
        raise ValueError("Uih and Vih must be (Ny, Nx, S) arrays with identical shapes.")
    Ny, Nx, S = Uih.shape
    N = Ny * Nx     # Number of nodes
    sqrt_c = np.sqrt((P.Lx * P.Ly) / float(Nx * Ny) ** 2)

    U = Uih.reshape(N, S)
    V = Vih.reshape(N, S)

    Phi = np.vstack([U.real, U.imag, V.real, V.imag])

    # Euclidean QR
    Q, R = np.linalg.qr(Phi, mode="reduced")  # Phi = Q R
    Q_do = Q / sqrt_c
    R_do = sqrt_c * R

    Ur = Q_do[:N, :]
    Ui = Q_do[N:2*N, :]
    Vr = Q_do[2*N:3*N, :]
    Vi = Q_do[3*N:, :]

    Uo = (Ur + 1j*Ui).reshape(Ny, Nx, S)
    Vo = (Vr + 1j*Vi).reshape(Ny, Nx, S)
    YY_new = YY @ R_do.T
    return YY_new, Uo, Vo


def orthonormalize_spectral(YY: np.ndarray,
                            Uarr: np.ndarray, Varr: np.ndarray,
                            P):
    """
    Gram–Schmidt in spectral inner product.
    """
    Ny, Nx, S = Uarr.shape

    Uih = Uarr.copy()
    Vih = Varr.copy()

    Uih0 = Uih.copy(); Vih0 = Vih.copy()
    # GS
    for i in range(S):
        nrm = np.sqrt(inner_product_spectral(Uih[..., i], Vih[..., i], Uih[..., i], Vih[..., i], P))
        Uih[..., i] /= nrm;  Vih[..., i] /= nrm
        for j in range(i+1, S):
            c = inner_product_spectral(Uih[..., j], Vih[..., j], Uih[..., i], Vih[..., i], P)
            Uih[..., j] -= c * Uih[..., i]
            Vih[..., j] -= c * Vih[..., i]

    # overlaps Over[i,j] = <old_j, new_i> (spectral)
    Over = np.zeros((S, S), dtype=float)
    for i in range(S):
        for j in range(S):
            Over[i, j] = inner_product_spectral(Uih0[..., j], Vih0[..., j], Uih[..., i], Vih[..., i], P)

    YY_new = YY @ Over.T

    return YY_new, Uih, Vih


def diag_cov_rotate(
    YY: np.ndarray,
    Uih: np.ndarray,
    Vih: np.ndarray,
    P: Params
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotate modes and coefficients to diagonalize CYY (descending eigenvalues), with sign-fix.

    Inputs
    ------
    YY  : (MC, S)
    Uih, Vih : (Ny, Nx, S)

    Returns
    -------
    YY1, U1, V1 : rotated/directed versions with CYY diagonal and modes energy-ordered.
    """
    MC, S = YY.shape
    if Uih.shape[-1] != S or Vih.shape[-1] != S:
        raise ValueError("Mode dimension mismatch between YY and (Uih,Vih).")


    # Covariance of YY
    CYY = second_moment(YY)
    w, V = np.linalg.eigh(np.real(CYY))      # ascending
    V = V[:, np.argsort(w)[::-1]]   # descending

    # # eigenvalues w are in ascending order from np.linalg.eigh
    # lam_min = w[0]
    # lam_max = w[-1]
    # # guard against lam_min == 0
    # cond = np.inf if lam_min <= 0 else lam_max / lam_min
    #
    # print("max|Y| =", np.max(np.abs(YY)))
    # print(f"lam_min = {lam_min:.3e}, lam_max = {lam_max:.3e}, cond(C) = {cond:.3e}")

    U0, V0 = Uih.copy(), Vih.copy()
    Uih = np.einsum("yxs,sp->yxp", Uih, V, optimize=True)
    Vih = np.einsum("yxs,sp->yxp", Vih, V, optimize=True)
    YY = YY @ V

    # Sign-fix: align new mode i with old mode i so inner product >= 0
    c = (P.Lx * P.Ly) / (P.Nx * P.Ny)**2
    prods = c * np.real((U0.conj() * Uih + V0.conj() * Vih).sum(axis=(0, 1)))
    signs = np.where(prods < 0.0, -1.0, 1.0)
    Uih *= signs[None, None, :]
    Vih *= signs[None, None, :]
    YY *= signs[None, :]
    return YY, Uih, Vih
