import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to npz result file")
    ap.add_argument("--out_dir", default="./save/DO_Lamb_Oseen/velocity_movies", help="Output movie directory")
    ap.add_argument("--r", type=int, help="Realisation index for DO reconstruction")
    ap.add_argument("--stride", type=int, default=4, help="Subsample stride for quiver")
    ap.add_argument("--arrwidth", type=float, default=0.01, help="arrow width for quiver (inches)")
    ap.add_argument("--step", type=int, default=2 , help="frame step (>1 to skip frames)")
    ap.add_argument("--fps", type=int, default=20, help="frames per second")
    ap.add_argument("--dpi", type=int, default=200, help="DPI for saved video")
    args = ap.parse_args()

    # load data, velocity fields, stochastic coefficients, and time values
    Z = np.load(args.input, allow_pickle=True)
    U  = Z["U_save"]        # (Ny, Nx, T)
    V  = Z["V_save"]
    Ui = Z["Ui_save"]       # (Ny, Nx, S, T)
    Vi = Z["Vi_save"]
    YY = Z["YY_save"]       # (MC, S, T)
    Tt = Z["T_save"]        # (T,)

    Ny, Nx, T = U.shape
    S = Ui.shape[2]
    MC = YY.shape[0]
    r = np.random.randint(0, MC) if args.r is None else int(np.clip(args.r, 0, MC-1))

    # grid
    dx = float(Z["Lx"]/Z["Nx"]); dy = float(Z["Ly"]/Z["Ny"])
    x = np.arange(0, float(Z["Lx"]), dx)[:Nx]
    y = np.arange(0, float(Z["Ly"]), dy)[:Ny]
    extent = [x.min(), x.max(), y.min(), y.max()]

    # --- quiver settings ---
    # plot one arrow every s gridpoints in both directions
    X, Y = np.meshgrid(x, y)
    s = slice(None, None, args.stride)
    Xs, Ys = X[s, s], Y[s, s]
    # find maximum magnitude of velocity field to scale arrows
    urec = U + np.einsum('xyst,st->xyt', Ui, YY[r])
    vrec = V + np.einsum('xyst,st->xyt', Vi, YY[r])
    M_mean_all = np.max(np.hypot(U, V))
    M_real_all = np.max(np.hypot(urec, vrec))
    M_global = max(M_mean_all, M_real_all) or 1.0
    Lscreen = 0.20 * min(Z["Lx"], Z["Ly"])  # largest arrow will occupy 20% of screen length
    scale = M_global / Lscreen

    # --- plotting ---
    plt.rcParams.update({"font.size": 14, "lines.linewidth": 1})
    fig = plt.figure(figsize=(17, 8), constrained_layout=True)

    # -- create axes --
    gs = GridSpec(2, 3, figure=fig, height_ratios=[3, 1])
    axs_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axs_top:
        ax.set_aspect("equal")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax_bottom = fig.add_subplot(gs[1, :])

    # -- heatmap colormap --
    cmap = plt.get_cmap("turbo") if "turbo" in plt.colormaps() else plt.get_cmap("viridis")
    norm = Normalize(vmin=0.0, vmax=M_global)

    # -- initial frame --
    # - left plot; mean -
    t0 = 0
    speed_mean0 = np.hypot(U[..., t0], V[..., t0])
    im_mean = axs_top[0].imshow(speed_mean0, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="bilinear")
    axs_top[0].set_title("Mean velocity $\\bar{u}$")
    Qmean = axs_top[0].quiver(
        Xs, Ys, U[s, s, t0], V[s, s, t0],
        color="k",
        angles="xy", scale_units="xy", scale=scale,
        units="inches", width=args.arrwidth, pivot="mid"
    )
    # - middle plot; whole realisation -
    Yrt0 = YY[r, :, t0]
    u0 = U[..., t0] + np.tensordot(Ui[..., :, t0], Yrt0, axes=(2, 0))
    v0 = V[..., t0] + np.tensordot(Vi[..., :, t0], Yrt0, axes=(2, 0))
    speed_real0 = np.hypot(u0, v0)
    im_real = axs_top[1].imshow(speed_real0, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="bilinear")
    axs_top[1].set_title(f"DO realisation r={r}: $\\bar{{u}}+\\sum_i Y_i\\,\\phi_i$")
    Qreal = axs_top[1].quiver(
        Xs, Ys, u0[s, s], v0[s, s],
        color="k",
        angles="xy", scale_units="xy", scale=scale,
        units="inches", width=args.arrwidth, pivot="mid"
    )
    # - right plot; modal contribution -
    du0, dv0 = u0 - U[..., t0], v0 - V[..., t0]
    speed_diff0 = np.hypot(du0, dv0)
    norm_diff = Normalize(vmin=0, vmax=0.35*M_global)
    im_diff = axs_top[2].imshow(speed_diff0, origin="lower", extent=extent,
                                cmap=cmap, norm=norm, interpolation="bilinear")
    axs_top[2].set_title(r"Difference $|u^{(r)}-\bar{u}|$")
    Qdiff = axs_top[2].quiver(
        Xs, Ys, du0[s, s], dv0[s, s],
        color="k",
        angles="xy", scale_units="xy", scale=scale,
        units="inches", width=args.arrwidth, pivot="mid"
    )

    # colorbars
    fig.colorbar(im_mean, ax=axs_top[0], fraction=0.046, pad=0.04).set_label(r"$|\mathbf{u}|$")
    fig.colorbar(im_real, ax=axs_top[1], fraction=0.046, pad=0.04).set_label(r"$|\mathbf{u}|$")
    fig.colorbar(im_diff, ax=axs_top[2], fraction=0.046, pad=0.04).set_label(r"$|Δ\mathbf{u}|$")

    lines = []
    for i in range(min(4, S)):
        (ln,) = ax_bottom.plot(Tt, YY[r, i, :], label=f"$Y_{i + 1}$")
        lines.append(ln)
    ax_bottom.set_xlim(Tt[0], Tt[-1])
    ax_bottom.set_xlabel("t")
    ax_bottom.set_ylabel("$Y_i$")
    ax_bottom.legend(frameon=False, ncol=min(4, S))
    time_marker = ax_bottom.axvline(Tt[0], color="k", ls="--", lw=1)

    # Frames to render
    frames = list(range(0, T, args.step))

    # ---------- Animator ----------
    def update(k):
        t = frames[k]
        # left (mean) panel
        im_mean.set_data(np.hypot(U[..., t], V[..., t]))
        Qmean.set_UVC(U[s, s, t], V[s, s, t])

        # middle (realisation) panel
        Yrt = YY[r, :, t]
        ur = U[..., t] + np.tensordot(Ui[..., :, t], Yrt, axes=(2, 0))
        vr = V[..., t] + np.tensordot(Vi[..., :, t], Yrt, axes=(2, 0))
        im_real.set_data(np.hypot(ur, vr))
        Qreal.set_UVC(ur[s, s], vr[s, s])

        # right (difference) panel
        du, dv = ur - U[..., t], vr - V[..., t]
        im_diff.set_data(np.hypot(du, dv))
        Qdiff.set_UVC(du[s, s], dv[s, s])

        # update Y_i inset
        time_marker.set_xdata([Tt[t]])

        fig.suptitle(f"t = {Tt[t]:.3f}")
        return (im_mean, Qmean, im_real, Qreal)

    # Ensure output dir exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Output filename
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output = os.path.join(args.out_dir, f"{base_name}.mp4")

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1000/args.fps, blit=False)
    writer = FFMpegWriter(
        fps=args.fps, codec="libx264", bitrate=2000,
        extra_args=["-pix_fmt", "yuv420p"]
    )
    update_func = lambda _i, _n: progress_bar.update(1)
    with tqdm(total=len(frames), desc="Saving video...") as progress_bar:
        ani.save(output, writer=writer, dpi=args.dpi, progress_callback=update_func)
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
