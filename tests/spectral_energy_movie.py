#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pathlib import Path

import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ----------------------------- helpers ----------------------------- #
def fft_wavenumber_magnitude(Nx, Ny, Lx, Ly):
    dx = Lx / Nx
    dy = Ly / Ny
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    return np.sqrt(KX * KX + KY * KY)


def make_radial_binner(K, nbins):
    Kmax = float(np.max(K))
    edges = np.linspace(0.0, Kmax, nbins + 1)
    k_cent = 0.5 * (edges[:-1] + edges[1:])

    flat = K.ravel()
    bin_id = np.searchsorted(edges, flat, side="right") - 1
    bin_id[(bin_id < 0) | (bin_id >= nbins)] = -1
    bin_id = bin_id.reshape(K.shape)

    return k_cent, bin_id


def radial_shell_sum(Ehat, bin_id, nbins):
    flat_id = bin_id.ravel()
    flat_E = Ehat.ravel()
    m = flat_id >= 0
    return np.bincount(flat_id[m], weights=flat_E[m], minlength=nbins)


def r_core(t):
    return np.sqrt(0.2 ** 2 + 0.1 * t)


def load_run(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    sigma = float(data["ou_amp"])
    t = np.array(data["T_save"], dtype=float)

    CYY = np.array(data["CYY_save"], dtype=float)  # (S+1, T)
    U_save = np.asarray(data["U_save"])            # (Ny, Nx, T)
    V_save = np.asarray(data["V_save"])
    Ui_save = np.asarray(data["Ui_save"])          # (Ny, Nx, S, T)
    Vi_save = np.asarray(data["Vi_save"])

    Lx = float(data["Lx"])
    Ly = float(data["Ly"])
    Ny, Nx, T = U_save.shape

    K = fft_wavenumber_magnitude(Nx, Ny, Lx, Ly)

    run = {
        "file": str(npz_path),
        "sigma": sigma,
        "t": t,
        "CYY": CYY,
        "U_save": U_save,
        "V_save": V_save,
        "Ui_save": Ui_save,
        "Vi_save": Vi_save,
        "Lx": Lx,
        "Ly": Ly,
        "Nx": Nx,
        "Ny": Ny,
        "K": K,
        "_radial_cache": {}
    }
    return run


def compute_Ek_snapshot(run, k0, nbins=40):
    Nx, Ny = run["Nx"], run["Ny"]
    fac = 1.0 / (Nx * Ny) ** 2

    U_save, V_save = run["U_save"], run["V_save"]
    Ui_save, Vi_save = run["Ui_save"], run["Vi_save"]
    lamY = run["CYY"][1:, :]  # (S, T) diag variances

    cache = run["_radial_cache"]
    if nbins in cache:
        k_cent, bin_id = cache[nbins]
    else:
        k_cent, bin_id = make_radial_binner(run["K"], nbins)
        cache[nbins] = (k_cent, bin_id)

    # mean spectrum
    uhat = np.fft.fft2(U_save[..., k0])
    vhat = np.fft.fft2(V_save[..., k0])
    Ehat_mean = 0.5 * (np.abs(uhat) ** 2 + np.abs(vhat) ** 2) * fac

    # stochastic spectrum (diag covariance)
    lam_k = lamY[:, k0]           # (S,)
    Ui_k = Ui_save[..., k0]       # (Ny, Nx, S)
    Vi_k = Vi_save[..., k0]

    Ui_hat = np.fft.fft2(Ui_k, axes=(0, 1))
    Vi_hat = np.fft.fft2(Vi_k, axes=(0, 1))

    var_u_hat = np.sum((np.abs(Ui_hat) ** 2) * lam_k[None, None, :], axis=2)
    var_v_hat = np.sum((np.abs(Vi_hat) ** 2) * lam_k[None, None, :], axis=2)
    Ehat_stoch = 0.5 * (var_u_hat + var_v_hat) * fac

    Em = radial_shell_sum(Ehat_mean, bin_id, nbins)
    Es = radial_shell_sum(Ehat_stoch, bin_id, nbins)
    Et = Em + Es

    return k_cent, Et, Em, Es


def sigma_folder_name(sigma):
    return f"ouamp-{sigma:g}"


def parse_inputs(inputs):
    parts = [p.strip() for p in inputs.split(",")]
    files = []
    for p in parts:
        if any(ch in p for ch in ["*", "?", "["]):
            files += list(Path().glob(p))
        else:
            files.append(Path(p))
    files = [f for f in files if f.suffix == ".npz"]
    files = sorted(files)
    if not files:
        raise FileNotFoundError("No .npz inputs found.")
    return files


def choose_frame_indices(t, tmin=None, tmax=None, stride=1):
    idx = np.arange(t.size, dtype=int)
    if tmin is not None:
        idx = idx[t >= float(tmin)]
    if tmax is not None:
        idx = idx[t <= float(tmax)]
    idx = idx[::max(1, int(stride))]
    if idx.size == 0:
        raise ValueError("No frames selected (check tmin/tmax/stride).")
    return idx


def plot_Ek_snapshot_frame(run, k0, outpng,
                           nbins=40, xscale="linear", yscale="log",
                           ic_band=(0.0, 8.0), forcing_band=(2.5, 8.5),
                           figsize=(8.2, 4.2), dpi=140):
    t0 = float(run["t"][k0])
    sigma = float(run["sigma"])

    k, Et, Em, Es = compute_Ek_snapshot(run, k0=k0, nbins=nbins)

    m = np.isfinite(k) & np.isfinite(Et) & (k >= 0)
    if xscale == "log":
        m &= (k > 0)
    if yscale == "log":
        m &= (Et > 0) & (Em > 0) & (Es > 0)

    k, Et, Em, Es = k[m], Et[m], Em[m], Es[m]

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.plot(k, Et, lw=2.2, color="k", label=r"$E_{\mathrm{tot}}(k)$")
    ax.plot(k, Em, lw=1.6, ls="--", color="lightseagreen", label=r"$E_{\mathrm{mean}}(k)$")
    ax.plot(k, Es, lw=1.6, ls="--", color="b", label=r"$E_{\mathrm{stoch}}(k)$")

    # shaded bands
    ax.axvspan(ic_band[0], ic_band[1], alpha=0.15, color="tab:orange", label="IC band")
    ax.axvspan(forcing_band[0], forcing_band[1], alpha=0.15, color="tab:green", label="forcing band")

    # k*(t) = 1 / r(t)
    kstar = 1.0 / r_core(t0)
    ax.axvline(kstar, lw=1.8, ls=":", color="tab:red", label=r"$k_{char}(t)=1/r(t)$")

    # ticks
    if xscale == "log":
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    # y-lims for log to avoid silly autoscale
    if yscale == "log" and Et.size:
        pos = Et[Et > 0]
        ymin = max(np.min(pos), 1e-12 * np.max(pos))
        ymax = np.max(pos)
        ax.set_ylim(ymin, 1.3 * ymax)

    ax.grid(True, alpha=0.2)
    ax.set_xlabel(r"$|k|$")
    ax.set_ylabel(r"$E(k)$ (shell-sum, snapshot)")
    ax.set_title(rf"$\sigma={sigma:g}$, $t={t0:.2f}$, $r(t)={r_core(t0):.3f}$, $k_{{char}}={kstar:.3f}$")
    ax.legend(frameon=False, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(outpng, bbox_inches="tight")
    plt.close(fig)


def run_ffmpeg(frames_dir, pattern, outmp4, fps, overwrite=True):
    frames_dir = Path(frames_dir)
    outmp4 = Path(outmp4)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
    ]
    if overwrite:
        cmd += ["-y"]
    cmd += [
        "-framerate", str(int(fps)),
        "-i", str(frames_dir / pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        # ensure even dims for H.264 compatibility
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(outmp4)
    ]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise RuntimeError('ffmpeg not found in PATH. Install it or make sure "ffmpeg" is callable.')


# ----------------------------- main ----------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Glob or comma-separated list of .npz files.")
    ap.add_argument("--outdir", default=None, help="Output directory (default: two levels above first input, then /plots).")

    ap.add_argument("--nbins", type=int, default=40)
    ap.add_argument("--xscale", type=str, default="linear", choices=["linear", "log"])
    ap.add_argument("--yscale", type=str, default="log", choices=["linear", "log"])

    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth time index as a frame.")
    ap.add_argument("--tmin", type=float, default=None)
    ap.add_argument("--tmax", type=float, default=None)

    ap.add_argument("--dpi", type=int, default=140)
    ap.add_argument("--figw", type=float, default=8.2)
    ap.add_argument("--figh", type=float, default=4.2)

    ap.add_argument("--ic_kmax", type=float, default=8.0)
    ap.add_argument("--forcing_kmin", type=float, default=2.5)
    ap.add_argument("--forcing_kmax", type=float, default=8.5)

    ap.add_argument("--keep_frames", action="store_true", help="Do not delete PNG frames after MP4 is created.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing MP4 / frames.")
    args = ap.parse_args()

    files = parse_inputs(args.inputs)
    outdir = Path(args.outdir) if args.outdir else (files[0].parents[1] / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    for f in files:
        run = load_run(f)
        sigdir = outdir / sigma_folder_name(run["sigma"])
        sigdir.mkdir(parents=True, exist_ok=True)

        frames_dir = sigdir / "Ek_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        outmp4 = sigdir / "Ek_snapshot_video.mp4"
        if outmp4.exists() and not args.overwrite:
            print(f"[skip] {outmp4} exists (use --overwrite to rebuild).")
            continue

        idx = choose_frame_indices(run["t"], tmin=args.tmin, tmax=args.tmax, stride=args.stride)
        pad = max(5, len(str(int(idx.size))))
        pattern = f"frame_%0{pad}d.png"

        # write frames
        if args.overwrite:
            for old in frames_dir.glob("frame_*.png"):
                old.unlink(missing_ok=True)

        print(f"[frames] {Path(run['file']).name}  sigma={run['sigma']:g}  -> {frames_dir}  ({idx.size} frames)")
        for j, k0 in tqdm(enumerate(idx)):
            outpng = frames_dir / f"frame_{j:0{pad}d}.png"
            plot_Ek_snapshot_frame(
                run,
                k0=int(k0),
                outpng=outpng,
                nbins=args.nbins,
                xscale=args.xscale,
                yscale=args.yscale,
                ic_band=(0.0, float(args.ic_kmax)),
                forcing_band=(float(args.forcing_kmin), float(args.forcing_kmax)),
                figsize=(args.figw, args.figh),
                dpi=args.dpi
            )

        # stitch video
        print(f"[ffmpeg] -> {outmp4}")
        run_ffmpeg(frames_dir, pattern, outmp4, fps=args.fps, overwrite=True)

        # cleanup
        if not args.keep_frames:
            for p in frames_dir.glob("frame_*.png"):
                p.unlink(missing_ok=True)
            try:
                frames_dir.rmdir()
            except OSError:
                pass

    print("Done. Saved under:", outdir)


if __name__ == "__main__":
    main()
