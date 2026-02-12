from __future__ import annotations
from .params import Params
import numpy as np


def build_grids(P: Params):
    """
    Return x, y, kx, ky, k2, dx, dy with NumPy FFT indexing.
    """
    dx = P.Lx / P.Nx
    dy = P.Ly / P.Ny
    x = np.linspace(0.0, P.Lx - dx, P.Nx, dtype=float)
    y = np.linspace(0.0, P.Ly - dy, P.Ny, dtype=float)

    kx_vec = (2 * np.pi / P.Lx) * np.concatenate(
        (np.arange(0, P.Nx // 2, dtype=float), np.arange(-P.Nx // 2, 0, dtype=float))
    )
    ky_vec = (2 * np.pi / P.Ly) * np.concatenate(
        (np.arange(0, P.Ny // 2, dtype=float), np.arange(-P.Ny // 2, 0, dtype=float))
    )
    kx, ky = np.meshgrid(kx_vec, ky_vec)
    k2 = kx**2 + ky**2
    return x, y, kx, ky, k2, dx, dy


def projection_multipliers(kx: np.ndarray, ky: np.ndarray):
    """
    p1 = ky^2/k^2, p2=-kx*ky/k^2, p3=kx^2/k^2; special case k=0.

    This Leray/Helmholtz projector removes the component parallel
    to k (the irrotational/pressure part) and keeps its orthogonal
    component.
    """
    k2 = kx**2 + ky**2
    p1 = np.zeros_like(kx, dtype=float)
    p2 = np.zeros_like(kx, dtype=float)
    p3 = np.zeros_like(kx, dtype=float)
    mask = k2 != 0.0
    invk2 = np.zeros_like(k2)
    invk2[mask] = 1.0 / k2[mask]
    p1[mask] = (ky[mask] * ky[mask]) * invk2[mask]
    p2[mask] = -(kx[mask] * ky[mask]) * invk2[mask]
    p3[mask] = (kx[mask] * kx[mask]) * invk2[mask]
    return p1, p2, p3