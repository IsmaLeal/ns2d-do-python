#!/usr/bin/env python3
"""
Replay the DOPlotter dashboard from a saved npz run and save it as a video.

This mimics the live ns2d_do.plotting.DOPlotter layout but drives it from
stored fields instead of during the solver loop.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib
# Force non-interactive backend and silence the live window calls inside DOPlotter
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
from tqdm import tqdm

from ns2d_do.plotting import DOPlotter
# Disable interactive calls inside DOPlotter
plt.show = lambda *args, **kwargs: None
plt.pause = lambda *args, **kwargs: None


def _fft_modes(ui_frame: np.ndarray, vi_frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return spectral versions of the spatial modes for one saved timestep."""
    Ny, Nx, S = ui_frame.shape
    Uih = np.empty((Ny, Nx, S), dtype=np.complex128)
    Vih = np.empty_like(Uih)
    for m in range(S):
        Uih[..., m] = np.fft.fft2(ui_frame[..., m])
        Vih[..., m] = np.fft.fft2(vi_frame[..., m])
    return Uih, Vih


def _default_output(npz_path: Path) -> Path:
    data_dir = npz_path.parent
    base_dir = data_dir.parent
    movies_dir = base_dir / "movies"
    movies_dir.mkdir(parents=True, exist_ok=True)
    return movies_dir / f"{npz_path.stem}_dashboard.mp4"


def autoscale_joint_scatter(ax, YY, pct=99.5, pad=0.08, symmetric=True):
    S = YY.shape[1]
    x = YY[:, 0]
    y = YY[:, 1] if S >= 2 else np.zeros_like(x)
    z = YY[:, 2] if S >= 3 else np.zeros_like(x)

    def bounds(d):
        d = np.asarray(d, dtype=float)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return (-1.0, 1.0)

        if symmetric:
            # pick a scale "a" so that ~pct% of points lie in [-a, a]
            a = float(np.nanpercentile(np.abs(d), pct))
            if a <= 0:
                a = float(np.max(np.abs(d))) if np.max(np.abs(d)) > 0 else 1e-12
            lo, hi = -a, a
        else:
            # pick lo/hi separately (can be off-centered)
            lo = float(np.nanpercentile(d, 100.0 - pct))
            hi = float(np.nanpercentile(d, pct))
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                c = float(np.nanmean(d)) if np.isfinite(np.nanmean(d)) else 0.0
                s = float(np.nanstd(d)) if np.nanstd(d) > 0 else 1e-12
                lo, hi = c - s, c + s

        span = hi - lo
        if span <= 0:
            span = max(abs(hi), 1.0)
        lo -= pad * span
        hi += pad * span
        return lo, hi

    xlo, xhi = bounds(x)
    ylo, yhi = bounds(y)
    zlo, zhi = bounds(z)

    ax.set_xlim3d(xlo, xhi)
    ax.set_ylim3d(ylo, yhi)
    ax.set_zlim3d(zlo, zhi)



def main() -> None:
    parser = argparse.ArgumentParser(description="Replay DOPlotter from a saved npz")
    parser.add_argument("--input", required=True, help="path to fields .npz written by main.py")
    parser.add_argument("--output", help="output video path (default: <run>/movies/<stem>_dashboard.mp4)")
    parser.add_argument("--sample", type=int, default=0, help="realisation index for the DOPlotter panel")
    parser.add_argument("--step", type=int, default=3, help="frame stride (>=1)")
    parser.add_argument("--fps", type=int, default=12, help="frames per second for the video")
    parser.add_argument("--dpi", type=int, default=120, help="DPI for the saved frames")
    parser.add_argument("--codec", default="libx264", help="FFmpeg codec (default libx264)")
    parser.add_argument("--bitrate", type=int, default=2000, help="FFmpeg bitrate")
    parser.add_argument("--quiv-step", type=int, default=4, help="stride for quiver arrows")
    parser.add_argument("--pad-plot", type=int, default=10, help="Fourier zero-padding factor (same as DOPlotter)")
    parser.add_argument("--cmap", default="turbo", help="matplotlib colormap name")
    args = parser.parse_args()

    run_path = Path(args.input).expanduser().resolve()
    data = np.load(run_path, allow_pickle=True)

    U = data["U_save"]  # (Ny, Nx, T)
    V = data["V_save"]
    Ui = data["Ui_save"]  # (Ny, Nx, S, T)
    Vi = data["Vi_save"]
    YY = data["YY_save"]  # (MC, S, T)
    Tt = data["T_save"]   # (T,)

    Ny, Nx, Nt = U.shape
    S = Ui.shape[2]
    MC = YY.shape[0]

    r = int(np.clip(args.sample, 0, MC - 1 if MC > 0 else 0))
    frames = list(range(0, Nt, max(1, args.step)))

    P = SimpleNamespace(
        Lx=float(data["Lx"]), Ly=float(data["Ly"]),
        Nx=int(Nx), Ny=int(Ny), Re=float(data["Re"]), S=int(S)
    )

    plt.ioff()
    plotter = DOPlotter(P, sample_index=r, quiv_step=args.quiv_step, pad_plot=args.pad_plot, cmap=args.cmap)

    output_path = Path(args.output) if args.output else _default_output(run_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = FFMpegWriter(fps=args.fps, codec=args.codec, bitrate=args.bitrate, extra_args=["-pix_fmt", "yuv420p"])
    fig = plotter.fig

    with writer.saving(fig, str(output_path), dpi=args.dpi):
        for idx in tqdm(frames, desc="Rendering frames"):
            Uh = np.fft.fft2(U[..., idx])
            Vh = np.fft.fft2(V[..., idx])
            Uih, Vih = _fft_modes(Ui[..., idx], Vi[..., idx])
            plotter.update(float(Tt[idx]), Uh, Vh, Uih, Vih, YY[..., idx])
            autoscale_joint_scatter(plotter.ax_scatter, YY[..., idx], pct=99.5, pad=0.08, symmetric=True)
            writer.grab_frame()

    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    main()
