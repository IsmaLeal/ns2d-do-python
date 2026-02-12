"""
Real-time plotting for the DO 2D NS solver.

Usage
-----
from ns2d_do.plotting import DOPlotter
plotter = DOPlotter(P)  # creates figures

Inside the solver's plot interval:
plotter.update(t, Uh, Vh, Uih, Vih, YY)

Notes
-----
- Uh, Vh, Uih, Vih are NON-shifted FFT arrays (np.fft layout), as in the solver.
- This computes vorticity spectrally: w = ifft2(i*kx*Vh - i*ky*Uh).real
- For quiver, we downsample the grid with a stride (default 3) to keep it snappy.
"""

from __future__ import annotations
import matplotlib as mpl
if "agg" in mpl.get_backend().lower():
    for cand in ("TkAgg", "Qt5Agg"):
        try:
            mpl.use(cand, force=True)
            break
        except Exception:
            pass
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from typing import Optional
import numpy as np
import math


def _pad_fourier(self, H, pad=2):
    Ny, Nx = H.shape
    My, Mx = pad*Ny, pad*Nx
    Hs = np.fft.fftshift(H)
    out = np.zeros((My, Mx), dtype=complex)
    cy, cx = My//2, Mx//2
    sy, sx = Ny//2, Nx//2
    out[cy-sy:cy+Ny-sy, cx-sx:cx+Nx-sx] = Hs   # drop-in center
    return np.fft.ifftshift(out)


def _vort_from_hat(self, Uh, Vh, kx, ky):
    ikx, iky = 1j*kx, 1j*ky
    wzh = ikx[None,:]*Vh - iky[:,None]*Uh
    return np.fft.ifft2(wzh).real


class DOPlotter:
    """
    Live dashboard:

      Row 1: Mean vorticity | Mode 1 vorticity | Mode 2 vorticity | Mode 3 vorticity
      Row 2: One realization vorticity | Mode 4 | Mode 5 | Mode 6
      Row 3: Joint distribution (Y1,Y2,Y3) | Variance vs time (log-y)

    Expected update(...) signature:
        update(t, Uh, Vh, Uih, Vih, YY, energy_mean=None, energies_modes=None)
    """

    def __init__(self, P, sample_index: int = 0, quiv_step: int = 4, cmap="turbo", pad_plot: int = 10):
        self.P = P
        self.sample_index = sample_index
        self.quiv_step = quiv_step
        self.cmap = cmap
        self.pad_plot = max(1, int(pad_plot))

        Ny, Nx = P.Ny, P.Nx
        self.x = np.linspace(0.0, P.Lx, Nx, endpoint=False)
        self.y = np.linspace(0.0, P.Ly, Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Wavenumbers for spectral vorticity and realization synthesis
        # Vorticity (curl) needs derivatives. These are easier to compute in
        # Fourier space, multiplying by i*k_x or i*k_y
        dx, dy = P.Lx / Nx, P.Ly / Ny
        self.kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
        self.ikx = 1j * self.kx[None, :]
        self.iky = 1j * self.ky[:, None]    # transpose one

        # Figure layout
        self.fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(3, 4, figure=self.fig, height_ratios=[1, 1, 0.9], hspace=0.55, wspace=0.35)

        # Axes grid
        self.ax_mean = self.fig.add_subplot(gs[0, 0])
        self.ax_modes = [
            self.fig.add_subplot(gs[0, 1]),
            self.fig.add_subplot(gs[0, 2]),
            self.fig.add_subplot(gs[0, 3]),
            self.fig.add_subplot(gs[1, 1]),
            self.fig.add_subplot(gs[1, 2]),
            self.fig.add_subplot(gs[1, 3]),
        ]
        self.ax_real = self.fig.add_subplot(gs[1, 0])
        self.ax_scatter = self.fig.add_subplot(gs[2, 0], projection="3d")
        self.ax_var = self.fig.add_subplot(gs[2, 1:4])

        # Images/contours/quivers/colorbars handles (created on first update)
        self._inited = False
        self.im_mean = None
        self.im_modes = [None] * 6
        self.im_real = None
        self.cb_mean = None
        self.cb_modes = [None] * 6
        self.cb_real = None
        self.q_mean = self.q_real = None
        self.q_modes = [None] * 6

        # Variance history
        self.t_hist = []
        self.var_lines = []  # matplotlib Line2D per mode

        # Titles/labels static
        self.ax_mean.set_title(f"Mean vorticity, t=0.00, Re={P.Re:.1f}")
        for i, ax in enumerate(self.ax_modes):
            ax.set_title(f"Mode {i+1} vorticity")
        self.ax_real.set_title("One realization vorticity (r=1)")
        self.ax_var.set_title("Variance")
        self.ax_var.set_xlabel("t")
        self.ax_var.set_ylabel(r"Var($Y_i$)")
        self.ax_var.set_yscale("log")
        self.ax_var.grid(True, which="both", ls=":")

        plt.show(block=False)
        plt.pause(0.001)

    # ---------- utilities ----------
    def _pad_center(self, H: np.ndarray) -> np.ndarray:
        """Center-embed spectrum H onto a (pad_plot*Ny, pad_plot*Nx) grid."""
        pad = self.pad_plot
        if pad == 1:
            return H
        Ny, Nx = H.shape
        My, Mx = pad * Ny, pad * Nx
        Hs = np.fft.fftshift(H)
        out = np.zeros((My, Mx), dtype=complex)
        cy, cx = My // 2, Mx // 2
        sy, sx = Ny // 2, Nx // 2
        out[cy - sy:cy - sy + Ny, cx - sx:cx - sx + Nx] = Hs
        return np.fft.ifftshift(out)

    def _upsample_field(self, Uh: np.ndarray, Vh: np.ndarray):
        """
        Zero-pad in Fourier **for plotting only** and return U,V,omega on the finer grid.
        Keeps physics exact (band-limited interpolation).
        """
        UhP = self._pad_center(Uh)
        VhP = self._pad_center(Vh)
        My, Mx = UhP.shape

        # fine-grid wavenumbers consistent with physical lengths
        kx_f = 2.0 * np.pi * np.fft.fftfreq(Mx, d=self.P.Lx / Mx)
        ky_f = 2.0 * np.pi * np.fft.fftfreq(My, d=self.P.Ly / My)
        ikx, iky = 1j * kx_f, 1j * ky_f

        # 2d curl in Fourier space
        wzh = ikx[None, :] * VhP - iky[:, None] * UhP
        U = np.fft.ifft2(UhP).real
        V = np.fft.ifft2(VhP).real
        W = np.fft.ifft2(wzh).real
        return U, V, W

    def _vorticity_from_hat(self, Uh, Vh):
        """ω = curl(u): ifft(i*kx*Vh - i*ky*Uh).real"""
        return np.fft.ifft2(self.ikx * Vh - self.iky * Uh).real

    def _make_panel(self, ax, field, title, with_quiver=False, u=None, v=None):
        My, Mx = field.shape
        x = np.linspace(0, self.P.Lx, Mx, endpoint=False)
        y = np.linspace(0, self.P.Ly, My, endpoint=False)
        Xf, Yf = np.meshgrid(x, y, indexing="xy")

        vmin, vmax = self._sym_clim(field)
        im = ax.imshow(
            field, origin="lower",
            extent=[0, self.P.Lx, 0, self.P.Ly],
            cmap=self.cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest", aspect="equal"
        )
        ax.contour(Xf, Yf, field, levels=10, colors="k", linewidths=0.3, alpha=0.35)
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        q = None
        if with_quiver and (u is not None) and (v is not None):
            qs = self.quiv_step
            q = ax.quiver(
                Xf[::qs, ::qs], Yf[::qs, ::qs],
                u[::qs, ::qs], v[::qs, ::qs],
                pivot="mid", scale=60, width=0.002, alpha=0.7
            )

        ax.set_xlim(0, self.P.Lx);
        ax.set_ylim(0, self.P.Ly)
        ax.set_xlabel("x");
        ax.set_ylabel("y")
        ax.set_title(title)
        return im, cb, q

    def _translation_correlations(self, Uh, Vh, Uih, Vih):
        """
        Project each mode onto the translation generators of the current mean:
            Gx = (∂Ū/∂x, ∂V̄/∂x),  Gy = (∂Ū/∂y, ∂V̄/∂y)
        Returns:
            cx, cy : arrays of shape (S,) with <Phi_i, Gx> and <Phi_i, Gy>
            tau    : shape (S,), magnitude sqrt(cx^2+cy^2)
            theta  : shape (S,), angle atan2(cy, cx) in radians
        Notes:
            - Inner product is the standard DO one in physical space up to a constant scale.
            - Any common scaling cancels in the angle.
        """
        # mean derivatives in spectral then to physical
        Ux = np.fft.ifft2((1j * self.kx[None, :]) * Uh).real
        Vx = np.fft.ifft2((1j * self.kx[None, :]) * Vh).real
        Uy = np.fft.ifft2((1j * self.ky[:, None]) * Uh).real
        Vy = np.fft.ifft2((1j * self.ky[:, None]) * Vh).real

        S = Uih.shape[2]
        cx = np.empty(S, dtype="float"); cy = np.empty(S, dtype="float")
        for i in range(S):
            Ui = np.fft.ifft2(Uih[..., i]).real
            Vi = np.fft.ifft2(Vih[..., i]).real
            cx[i] = np.mean(Ui * Ux + Vi * Vx)
            cy[i] = np.mean(Ui * Uy + Vi * Vy)

        dx = -cx; dy = -cy
        tau = np.hypot(dx, dy)
        theta = np.rad2deg(np.arctan2(dy, dx))
        return cx, cy, tau, theta

    def _update_translational_arrows(self, thetas, taus=None):
        """
        Draw or update two arrows on the 1-realisation panel showing
        the directions of the most translation-like modes.
        """
        if thetas is None or len(thetas) == 0:
            return
        # two anchors near the top-left
        x0 = 0.5 * self.P.Lx
        y0 = 0.5 * self.P.Ly
        L = 0.25 * min(self.P.Lx, self.P.Ly)  # arrow length for display

        if taus is None:
            Ls = np.full(min(2, len(thetas)), L)
        else:
            t2 = np.asarray(taus[:2], dtype=float)
            m = float(np.max(t2)) if np.max(t2) > 0 else 1.0
            Ls = L * (t2 / m)**0.7

        th = np.asarray(thetas[:2], dtype=float)
        X = np.array([x0, x0][:len(th)])
        Y = np.array([y0, y0][:len(th)])
        U = Ls[:len(th)] * np.cos(th)
        V = Ls[:len(th)] * np.sin(th)

        if getattr(self, "_q_trans", None) is None:
            # one quiver with two vectors, fixed length in data units
            self._q_trans = self.ax_real.quiver(
                X, Y, U, V,
                color=["tab:red", "tab:purple"][:len(th)],
                angles="xy", scale_units="xy", scale=1.0,
                width=0.01, alpha=0.9, pivot="tail"
            )
        else:
            self._q_trans.set_offsets(np.c_[X, Y])
            self._q_trans.set_UVC(U, V)

    @staticmethod
    def _sym_clim(field, pct=99.5):
        a = np.percentile(np.abs(field), pct)
        a = float(a) if a > 0 else float(np.max(np.abs(field)))
        return -a, a

    @staticmethod
    def _soft_set_clim(im, new_vmin, new_vmax, alpha=0.2):
        vmin, vmax = im.get_clim()
        im.set_clim(vmin * (1 - alpha) + new_vmin * alpha,
                    vmax * (1 - alpha) + new_vmax * alpha)

    # ---------- public API ----------
    def update(self, t, Uh, Vh, Uih, Vih, YY, energy_mean=None, energies_modes=None):
        """
        Update all panels using spectral inputs.
        Upsamples for plotting only (zero-padded Fourier) to get smoother maps.
        """
        S = Uih.shape[2]
        M = min(6, S)

        # --- Mean (upsampled) ---
        U_mean, V_mean, W_mean = self._upsample_field(Uh, Vh)

        # --- Modes (upsampled) ---
        om_modes = []
        Ui_fine, Vi_fine = [], []
        for i in range(M):
            U_f, V_f, W_f = self._upsample_field(Uih[..., i], Vih[..., i])
            Ui_fine.append(U_f); Vi_fine.append(V_f); om_modes.append(W_f)

        # --- One realisation (upsampled) ---
        r = int(np.clip(self.sample_index, 0, YY.shape[0] - 1)) # select random row (realisation)
        # velocities are mean + sum(u_i * Y_i)
        comboU_hat = Uh + np.tensordot(Uih, YY[r, :], axes=(2, 0))
        comboV_hat = Vh + np.tensordot(Vih, YY[r, :], axes=(2, 0))
        Ur_f, Vr_f, Wr_f = self._upsample_field(comboU_hat, comboV_hat)

        # Translation-dominant directions from mean & modes (coarse grid; angle unaffected)
        cx, cy, tau, theta = self._translation_correlations(Uh, Vh, Uih, Vih)
        order = np.argsort(tau)[::-1]  # largest two
        top = order[:2] if tau.size >= 2 else order[:1]
        thetas_two = theta[top] if top.size > 0 else None   # already in degrees
        taus_two = tau[top] if top.size > 0 else None

        if thetas_two is not None and thetas_two.size > 0:
            tag = " | " + ", ".join([f"θ{i+1}={deg:.0f}°" for i, deg in enumerate(thetas_two[:2])])
        else:
            tag = ""

        # --- First draw: create artists ---
        if not self._inited:
            # Mean
            self.im_mean, self.cb_mean, self.q_mean = self._make_panel(
                self.ax_mean, W_mean, f"Mean vorticity, t={t:.2f}, Re={self.P.Re:.1f}",
                with_quiver=True, u=U_mean, v=V_mean
            )
            # Modes (up to 6)
            for i in range(M):
                self.im_modes[i], self.cb_modes[i], self.q_modes[i] = self._make_panel(
                    self.ax_modes[i], om_modes[i], f"Mode {i + 1} vorticity",
                    with_quiver=True, u=Ui_fine[i], v=Vi_fine[i]
                )
            # Realization
            self.im_real, self.cb_real, self.q_real = self._make_panel(
                self.ax_real, Wr_f, f"One realization vorticity (r={r + 1}){tag}",
                with_quiver=True, u=Ur_f, v=Vr_f
            )
            self._update_translational_arrows(np.deg2rad(thetas_two), taus_two)

            # Joint distribution + variance
            self._init_scatter(YY)
            self._init_var_lines(S)
            self._inited = True
            plt.pause(0.001)
            return

        # --- Updates (data + clims + quiver) ---
        self.im_mean.set_data(W_mean)
        vmin, vmax = self._sym_clim(W_mean); self._soft_set_clim(self.im_mean, vmin, vmax)
        self.cb_mean.update_normal(self.im_mean)
        if self.q_mean is not None:
            qs = self.quiv_step
            self.q_mean.set_UVC(U_mean[::qs, ::qs], V_mean[::qs, ::qs])
        self.ax_mean.set_title(f"Mean vorticity, t={t:.2f}, Re={self.P.Re:.1f}")

        for i in range(M):
            self.im_modes[i].set_data(om_modes[i])
            vmin, vmax = self._sym_clim(om_modes[i]); self._soft_set_clim(self.im_modes[i], vmin, vmax)
            self.cb_modes[i].update_normal(self.im_modes[i])
            if self.q_modes[i] is not None:
                qs = self.quiv_step
                self.q_modes[i].set_UVC(Ui_fine[i][::qs, ::qs], Vi_fine[i][::qs, ::qs])

        self.im_real.set_data(Wr_f)
        vmin, vmax = self._sym_clim(Wr_f); self._soft_set_clim(self.im_real, vmin, vmax)
        self.cb_real.update_normal(self.im_real)
        if self.q_real is not None:
            qs = self.quiv_step
            self.q_real.set_UVC(Ur_f[::qs, ::qs], Vr_f[::qs, ::qs])
        self.ax_real.set_title(f"One realisation vorticity (r={r+1}){tag}")
        self._update_translational_arrows(np.deg2rad(thetas_two), taus_two)

        # Scatter & variance
        self._update_scatter(YY)
        self._update_variance(t, YY)

        plt.pause(0.001)

    # ---------- scatter (Y1,Y2,Y3) ----------

    def _init_scatter(self, YY):
        self.ax_scatter.cla()
        S = YY.shape[1]
        self.ax_scatter.set_title("Joint distribution")
        if S >= 3:
            self._sc = self.ax_scatter.scatter(YY[:, 0], YY[:, 1], YY[:, 2], s=8, alpha=0.5)
            self.ax_scatter.set_xlabel("Y2"); self.ax_scatter.set_ylabel("Y1"); self.ax_scatter.set_zlabel("Y3")
        elif S == 2:
            self._sc = self.ax_scatter.scatter(YY[:, 0], YY[:, 1], np.zeros_like(YY[:, 0]), s=8, alpha=0.5)
            self.ax_scatter.set_xlabel("Y1"); self.ax_scatter.set_ylabel("Y2"); self.ax_scatter.set_zlabel("")
        else:
            self._sc = None
            self.ax_scatter.text2D(0.1, 0.5, "S<2, no scatter", transform=self.ax_scatter.transAxes)

    def _update_scatter(self, YY):
        if getattr(self, "_sc", None) is None:
            return
        S = YY.shape[1]
        if S >= 3:
            self._sc._offsets3d = (YY[:, 0], YY[:, 1], YY[:, 2])
        elif S == 2:
            self._sc._offsets3d = (YY[:, 0], YY[:, 1], np.zeros_like(YY[:, 0]))

    # ---------- variance panel ----------

    def _init_var_lines(self, S):
        self.var_lines = []
        for i in range(S):
            (line,) = self.ax_var.semilogy([], [], label=f"Mode {i+1}")
            self.var_lines.append(line)
        self.ax_var.legend(ncol=S, fontsize=8, frameon=False)

    def _update_variance(self, t, YY):
        # Var(Y_i) with population normalization (consistent with second_moment)
        MC = YY.shape[0]
        CYY = (YY.T @ YY) / MC
        diag = np.diag(CYY)
        self.t_hist.append(float(t))
        for i, line in enumerate(self.var_lines):
            ydata = list(line.get_ydata())
            ydata.append(float(diag[i]) if i < diag.size else 0.0)
            xdata = list(line.get_xdata()); xdata.append(float(t))
            line.set_data(xdata, ydata)
        # keep axes tidy
        self.ax_var.relim(); self.ax_var.autoscale_view()