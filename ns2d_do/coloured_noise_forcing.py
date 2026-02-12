from __future__ import annotations
import numpy as np

from .grids import build_grids
from .params import Params


class OU:
    """
    Ornstein-Uhlenbeck scalar with exact mean/variance discretisation.
        x_{n+1} = alpha * x_n + sqrt(var * (1 - alpha^2)) * N(0,1),
    where alpha = exp(-dt/tau) and var is the stationary variance.
    """
    def __init__(
            self,
            dt: float,
            mu: float,
            tau: float,
            var: float,
            rng: np.random.Generator):
        if tau <= 0.0:
            raise ValueError("OU 'tau' must be > 0.")
        if var < 0.0:
            raise ValueError("OU 'var' must be >= 0.")
        self.alpha = float(np.exp(-dt / tau))
        self.scale = float(np.sqrt(var * (1.0 - self.alpha ** 2)))
        self.x = 0.0
        self.rng = rng
        self.mu = float(mu)

    def step(self):
        self.x = self.alpha * self.x + self.scale * self.rng.normal()
        return self.mu + self.x


def _canonical_half_ring(k_pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Keep exactly one representative from each ±k pair (the 'half ring').
    Rule: (ky>0) or (ky==0 and kx>0).
    A real physical field imposes Hermitian symmetry: Fh(−k) = Fh(k). If we
    directly iterate over all k including negatives and also add conjugates, we’d
    double-count. Selecting a half ring gives one representative per pair; we then
    explicitly set the conjugate at −k once.
    """
    half = []
    for kx, ky in k_pairs:
        if ky > 0 or (ky == 0 and kx > 0):
            half.append((kx, ky))
    # deduplicate
    half = sorted(list(set(half)))
    return half


class SpectralOUForcing:
    """
    Band-limited coloured forcing in spectral space on a k-band.
    Returns (Fx_hat, Fy_hat) each step, already divergence-free at the forced modes.
    """

    def __init__(self, P, seed_offset: int = 0):
        self.P = P
        self.dt = P.dt
        self.nx, self.ny = P.Nx, P.Ny
        self.Lx, self.Ly = P.Lx, P.Ly
        x, y, kx, ky, k2, dx, dy = build_grids(P)
        self.kx = kx; self.ky = ky; self.k2 = k2

        k_mag = np.sqrt(self.k2)
        k_min = max(0.0, float(P.ou_k_min))
        k_max = float(P.ou_k_max)
        if k_max <= 0.0 or k_max < k_min:
            raise ValueError("Invalid k-band: ensure ou_k_max > ou_k_min >= 0.")

        mask = (k_mag >= k_min) & (k_mag <= k_max) & (k_mag > 0.0)
        # Convert physical wavenumbers back to integer lattice indices
        iy_ix = np.argwhere(mask)
        k_pairs: list[tuple[int, int]] = []
        for iy, ix in iy_ix:
            kx_idx = int(np.round(self.kx[iy, ix] * self.Lx / (2.0 * np.pi)))
            ky_idx = int(np.round(self.ky[iy, ix] * self.Ly / (2.0 * np.pi)))
            if kx_idx == 0 and ky_idx == 0:
                continue
            k_pairs.append((kx_idx, ky_idx))

        self.half = _canonical_half_ring(k_pairs)
        if not self.half:
            raise ValueError("No band modes selected; adjust ou_k_min/ou_k_max.")

        self.rng = np.random.default_rng(int(P.Seed)+int(seed_offset))
        # 0 mean because here the OU noise is additive
        self.ou = [OU(self.dt, 0, P.ou_tau, P.ou_var, self.rng) for _ in self.half]
        self.amp = float(P.ou_amp)

        # pre-alloc spectral arrays
        self._Fxh = np.zeros((self.ny, self.nx), dtype=np.complex128)
        self._Fyh = np.zeros((self.ny, self.nx), dtype=np.complex128)

    @staticmethod
    def _wrap(ix: int, n: int) -> int:
        return ix % n

    def step_hat(self) -> tuple[np.ndarray, np.ndarray]:
        Fxh = self._Fxh; Fyh = self._Fyh
        Fxh[:] = 0.0; Fyh[:] = 0.0

        # For each representative +k in the half ring:
        #   1) draw OU amplitude 'a'
        #   2) build divergence-free vector at +k along e_perp = (-ky, kx)/|k|
        #   3) set F(+k) = a * e_perp
        #   4) enforce Hermitian symmetry: F(-k) = conj(F(+k))  (so ifft is real)
        for (ou, (kx_idx, ky_idx)) in zip(self.ou, self.half):
            a = self.amp * ou.step()

            ixp = self._wrap(kx_idx, self.nx)
            iyp = self._wrap(ky_idx, self.ny)
            ixn = self._wrap(-kx_idx, self.nx)
            iyn = self._wrap(-ky_idx, self.ny)

            # physical wavenumber components at +k from existing meshgrids
            kx = self.kx[iyp, ixp]
            ky = self.ky[iyp, ixp]
            k2 = self.k2[iyp, ixp]
            if k2 == 0.0:
                # shouldn't happen for ring modes, but guard anyway
                continue

            invk = 1.0 / np.sqrt(k2)
            ex = -ky * invk  # x-component of e_perp
            ey =  kx * invk  # y-component of e_perp
            Fx_k = a * ex
            Fy_k = a * ey

            # place at +k
            Fxh[iyp, ixp] += Fx_k
            Fyh[iyp, ixp] += Fy_k

            # enforce real field via Hermitian symmetry
            Fxh[iyn, ixn] += np.conj(Fx_k)
            Fyh[iyn, ixn] += np.conj(Fy_k)

        # zero mean (optional)
        Fxh[0, 0] = 0.0
        Fyh[0, 0] = 0.0
        return Fxh, Fyh


def make_forcing(P):
    """
    Factory: returns an object with .step_hat() or None.
    P must carry: forcing_type, ou_tau, ou_var, ou_amp, ou_k_min, ou_k_max, and kx, ky, k2 grids.
    """
    ftype = getattr(P, "forcing_stochastic", "none")
    if ftype == "spectral_ou":
        return [SpectralOUForcing(P, seed_offset=s) for s in range(P.MC)]
    return None
