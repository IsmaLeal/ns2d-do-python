from __future__ import annotations
from typing import Tuple
import numpy as np


def _zero_nyquist_lines(A):
    Ny, Nx = A.shape
    if Nx % 2 == 0:
        A[:, Nx//2] = 0
    if Ny % 2 == 0:
        A[Ny//2, :] = 0
    return A


def _pad_spectrum_3_2(A: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """
    Zero-pad a non-fftshifted 2D spectrum A (Ny,Nx) to (3Ny/2, 3Nx/2)
    by inserting zeros in the middle of each axis.

    Layout assumption matches np.fft:
      columns: [0, 1, ..., hNx-1, -hNx, ..., -1]
      rows:    [0, 1, ..., hNy-1, -hNy, ..., -1]
    """
    Ny, Nx = A.shape
    hNy, hNx = Ny // 2, Nx // 2
    My, Mx = Ny + hNy, Nx + hNx  # == 3/2 sizes

    if out is None:
        out = np.zeros((My, Mx), dtype=A.dtype)
    else:
        out.fill(0)

    # Quadrants (top/bottom x left/right):
    # Horizontal: [left half]  zeros  [right half]
    # Vertical:   [top half]   zeros  [bottom half]

    # Top-left
    out[0:hNy, 0:hNx] = A[0:hNy, 0:hNx]
    # Top-right
    out[0:hNy, Mx - hNx: Mx] = A[0:hNy, hNx:Nx]
    # Bottom-left
    out[My - hNy: My, 0:hNx] = A[hNy:Ny, 0:hNx]
    # Bottom-right
    out[My - hNy: My, Mx - hNx: Mx] = A[hNy:Ny, hNx:Nx]

    return out


def _crop_spectrum_3_2(B: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """
    Crop a (3Ny/2, 3Nx/2) spectrum back to (Ny, Nx), taking the four
    corner blocks and multiplying by (3/2)^2 to compensate normalization.
    """
    My, Mx = B.shape
    Ny = (2 * My) // 3
    Nx = (2 * Mx) // 3
    hNy, hNx = Ny // 2, Nx // 2


    if out is None:
        out = np.zeros((Ny, Nx), dtype=B.dtype)
    else:
        out.fill(0)

    # Top-left
    out[0:hNy, 0:hNx] = B[0:hNy, 0:hNx]
    # Top-right
    out[0:hNy, hNx:Nx] = B[0:hNy, Mx - hNx: Mx]
    # Bottom-left
    out[hNy:Ny, 0:hNx] = B[My - hNy: My, 0:hNx]
    # Bottom-right
    out[hNy:Ny, hNx:Nx] = B[My - hNy: My, Mx - hNx: Mx]

    # Scale factor = (Mx/Nx) * (My/Ny) = (3/2) * (3/2)
    out *= (Mx / Nx) * (My / Ny)
    return out



def aa_product(Uh: np.ndarray, Vh: np.ndarray, ws=None) -> np.ndarray:
    """
    Anti-aliased spectral product of two scalar fields whose spectra are Uh, Vh.
    Implements the 3/2-rule:
      1) pad spectra to 3/2 size,
      2) ifft to physical space,
      3) pointwise multiply,
      4) fft back,
      5) crop corners and scale by (3/2)^2.

    Parameters
    ----------
    Uh, Vh : (Ny, Nx) complex
        Non-shifted Fourier coefficients (np.fft layout).

    Returns
    -------
    UVh : (Ny, Nx) complex
        Anti-aliased spectrum of the product u*v.
    """
    if ws is None:
        # 1) pad
        Uh_pad = _pad_spectrum_3_2(_zero_nyquist_lines(Uh.copy()))
        Vh_pad = _pad_spectrum_3_2(_zero_nyquist_lines(Vh.copy()))

        # 2) to physical
        u = np.fft.ifft2(Uh_pad)
        v = np.fft.ifft2(Vh_pad)

        # 3) product in physical space and back to spectral
        Wh = np.fft.fft2(u * v)
        return _crop_spectrum_3_2(Wh)

    Uh_pad, Vh_pad, out = ws
    Uh_pad = _pad_spectrum_3_2(_zero_nyquist_lines(Uh.copy()), out=Uh_pad)
    Vh_pad = _pad_spectrum_3_2(_zero_nyquist_lines(Vh.copy()), out=Vh_pad)
    u = np.fft.ifft2(Uh_pad)
    v =  np.fft.ifft2(Vh_pad)
    Wh = np.fft.fft2(u * v)
    return _crop_spectrum_3_2(Wh, out=out)


def project_div_free(
        Fxh: np.ndarray,
        Fyh: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helmholtz/Leray projection in Fourier space:
      [PFx; PFy] = [[p1, p2],[p2, p3]] * [Fx; Fy], applied pointwise in k-space.
    Leaves the (0,0) mode unchanged (mean pressure ambiguity).
    """
    if Fxh.shape != Fyh.shape or Fxh.shape != p1.shape:
        raise ValueError("All arrays must have identical shapes.")
    PFx = p1 * Fxh + p2 * Fyh
    PFy = p2 * Fxh + p3 * Fyh
    PFx[0, 0] = Fxh[0, 0]
    PFy[0, 0] = Fyh[0, 0]
    return PFx, PFy


def enforce_hermitian_symmetry(F):
    Ny, Nx = F.shape
    for j in range(Ny):
        jj = (-j) % Ny
        for i in range(Nx):
            ii = (-i) % Nx
            # visit each conjugate pair once
            if (j > jj) or (j == jj and i > ii):
                continue
            avg = 0.5 * (F[j, i] + np.conj(F[jj, ii]))
            F[j, i] = avg
            F[jj, ii] = np.conj(avg)
    # Nyquist lines must be real for even sizes
    if Nx % 2 == 0:
        F[:, Nx // 2] = F[:, Nx // 2].real
    if Ny % 2 == 0:
        F[Ny // 2, :] = F[Ny // 2, :].real
    F[0, 0] = F[0, 0].real
    return F