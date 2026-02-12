from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, argparse
import numpy as np

def rob_scale(a, pct=99.0):
    a = np.asarray(a)
    if not a.size:
        return 1.0
    s = float(np.percentile(np.abs(a), pct))
    return s if s > 0 else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .npz result file")
    ap.add_argument("--out_dir", default="./save/DO_Lamb_Oseen/vorticity_movies")
    ap.add_argument("--r", type=int, help="realisation index")
    ap.add_argument("--stride", type=int, default=4, help="stride for quiver")
    ap.add_argument("--arrwidth", type=float, default=0.01, help="arrow width for quiver (inches)")
    ap.add_argument("--step", type=int, default=2, help="frame step (>1 to skip frames)")
    ap.add_argument("--fps", type=int, default=20, help="frames per second")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    # load data
    Z  = np.load(args.input, allow_pickle=True)
    U  = Z["U_save"]        # (Ny,Nx,T)
    V  = Z["V_save"]
    Ui = Z["Ui_save"]       # (Ny,Nx,S,T)
    Vi = Z["Vi_save"]
    YY = Z["YY_save"]       # (MC,S,T)
    W  = Z["W_save"]        # (Ny, Nx, T)
    Wi = Z["Wi_save"]       # (Ny, Nx, S, T)
    Tt = Z["T_save"]        # (T,)

    Lx, Ly = float(Z["Lx"]), float(Z["Ly"])
    Ny, Nx, T = U.shape
    S = Ui.shape[2]
    MC = YY.shape[0]
    r = np.random.randint(0, MC) if args.r is None else int(np.clip(args.r, 0, MC-1))

    # precompute variances Var(Y_i)(t) across realisations
    # YY has shape (MC, S, T) -> var_YY is (S, T)
    var_YY = YY.var(axis=0)  # population variance (ddof=0)

    # grid
    dx = Lx / Nx
    dy = Ly / Ny
    x = np.arange(0, Lx, dx)[:Nx]
    y = np.arange(0, Ly, dy)[:Ny]
    extent = [x.min(), x.max(), y.min(), y.max()]

    # --- quiver settings ---
    X, Y = np.meshgrid(x, y)
    s = slice(None, None, args.stride)
    Xs, Ys = X[s, s], Y[s, s]

    # global arrow scale from max speed over time (mean + one realisation)
    urec = U + np.einsum("xyst,st->xyt", Ui, YY[r])
    vrec = V + np.einsum("xyst,st->xyt", Vi, YY[r])
    M_global = max(np.max(np.hypot(U, V)), np.max(np.hypot(urec, vrec))) or 1.0
    Lscreen = 0.20 * min(Lx, Ly)  # longest arrow ~20% of shorter box side
    scale = M_global / Lscreen

    # --- plotting ---
    plt.rcParams.update({"font.size": 14, "lines.linewidth": 1})
    fig = plt.figure(figsize=(17, 10), constrained_layout=True)

    # axes
    gs  = GridSpec(3, 3, figure=fig, height_ratios=[2, 2, 1])
    ax_mean = fig.add_subplot(gs[0, 0])
    ax_m1   = fig.add_subplot(gs[0, 1])
    ax_m2   = fig.add_subplot(gs[0, 2])
    ax_real = fig.add_subplot(gs[1, 0])
    ax_m3   = fig.add_subplot(gs[1, 1])
    ax_m4   = fig.add_subplot(gs[1, 2])
    ax_Y    = fig.add_subplot(gs[2, :])

    for ax in [ax_mean, ax_m1, ax_m2, ax_real, ax_m3, ax_m4]:
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    # colormap and fixed symmetric norm (for scaled fields)
    cmap = plt.get_cmap("turbo") if "turbo" in plt.colormaps() else plt.get_cmap("seismic")
    norm01 = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    # --- initial fields (t0) ---
    t0 = 0
    w_mean0 = W[..., t0]

    Yrt0 = YY[r, :, t0]
    u0 = U[..., t0] + np.tensordot(Ui[..., :, t0], Yrt0, axes=(2, 0))
    v0 = V[..., t0] + np.tensordot(Vi[..., :, t0], Yrt0, axes=(2, 0))
    w_real0 = W[..., t0] + np.tensordot(Wi[..., :, t0], Yrt0, axes=(2, 0))

    w1 = Wi[..., 0, t0] if S >= 1 else None
    w2 = Wi[..., 1, t0] if S >= 2 else None
    w3 = Wi[..., 2, t0] if S >= 3 else None
    w4 = Wi[..., 3, t0] if S >= 4 else None

    # helper: one panel
    def add_panel(ax, w_data, u_data, v_data, title, cbar_label):
        im = ax.imshow(
            w_data,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm01,
            interpolation="bilinear",
        )
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        quiv = ax.quiver(
            Xs,
            Ys,
            u_data[s, s],
            v_data[s, s],
            color="k",
            angles="xy",
            scale_units="xy",
            scale=scale,
            units="inches",
            width=args.arrwidth,
            pivot="mid",
        )
        return im, cbar, quiv

    # initial robust scale s(t0) across visible panels
    s_flow0 = max(rob_scale(w_mean0), rob_scale(w_real0)) or 1.0

    w_modes = [w1, w2, w3, w4]
    s_modes = [
        rob_scale(w) if (w is not None and i < S) else 1.0
        for i, w in enumerate(w_modes)
    ]
    s_m1, s_m2, s_m3, s_m4 = s_modes

    # draw initial scaled images
    im_mean, cb_mean, q_mean = add_panel(
        ax_mean,
        w_mean0 / s_flow0,
        U[..., t0],
        V[..., t0],
        r"Mean vorticity $\bar{\omega}$",
        r"$\omega/s(t)$",
    )
    im_real, cb_real, q_real = add_panel(
        ax_real,
        w_real0 / s_flow0,
        u0,
        v0,
        f"DO realisation r={r}: $\\omega^{{(r)}}$",
        r"$\omega/s(t)$",
    )
    cb_mean.set_label(r"$\omega/s_{\rm flow}(t)$")
    cb_real.set_label(r"$\omega/s_{\rm flow}(t)$")
    if S >= 1:
        im_m1, cb_m1, q_m1 = add_panel(
            ax_m1,
            w1 / s_m1,
            Ui[..., 0, t0],
            Vi[..., 0, t0],
            r"Mode 1 vorticity $\omega_1$",
            r"$\omega_1/s(t)$",
        )
        cb_m1.set_label(r"$\omega_1/s_{\rm modes}(t)$")
    if S >= 2:
        im_m2, cb_m2, q_m2 = add_panel(
            ax_m2,
            w2 / s_m2,
            Ui[..., 1, t0],
            Vi[..., 1, t0],
            r"Mode 2 vorticity $\omega_2$",
            r"$\omega_2/s(t)$",
        )
        cb_m2.set_label(r"$\omega_2/s_{\rm modes}(t)$")
    if S >= 3:
        im_m3, cb_m3, q_m3 = add_panel(
            ax_m3,
            w3 / s_m3,
            Ui[..., 2, t0],
            Vi[..., 2, t0],
            r"Mode 3 vorticity $\omega_3$",
            r"$\omega_3/s(t)$",
        )
        cb_m3.set_label(r"$\omega_3/s_{\rm modes}(t)$")
    if S >= 4:
        im_m4, cb_m4, q_m4 = add_panel(
            ax_m4,
            w4 / s_m4,
            Ui[..., 3, t0],
            Vi[..., 3, t0],
            r"Mode 4 vorticity $\omega_4$",
            r"$\omega_4/s(t)$",
        )
        cb_m4.set_label(r"$\omega_4/s_{\rm modes}(t)$")

    # bottom plot: Var(Y_i)(t) for first 4 modes
    lines = []
    for i in range(min(4, S)):
        (ln,) = ax_Y.plot(Tt, var_YY[i, :], label=fr"$\mathrm{{Var}}(Y_{i+1})$")
        lines.append(ln)
    ax_Y.set_xlim(Tt[0], Tt[-1])
    ax_Y.set_xlabel("t")
    ax_Y.set_ylabel(r"$\mathrm{Var}(Y_i)$")
    ax_Y.set_yscale("log")
    if S > 0:
        ax_Y.legend(frameon=False, ncol=min(4, S))
    time_marker = ax_Y.axvline(Tt[0], color="k", ls="--", lw=1)

    frames = list(range(0, T, args.step))

    def update(k):
        t = frames[k]

        # fields
        w_mean = W[..., t]

        Yrt = YY[r, :, t]
        ur = U[..., t] + np.tensordot(Ui[..., :, t], Yrt, axes=(2, 0))
        vr = V[..., t] + np.tensordot(Vi[..., :, t], Yrt, axes=(2, 0))
        w_real = W[..., t] + np.tensordot(Wi[..., :, t], Yrt, axes=(2, 0))

        w_list = [w_mean, w_real]
        w1 = w2 = w3 = w4 = None
        if S >= 1:
            w1 = Wi[..., 0, t]
            w_list.append(w1)
        if S >= 2:
            w2 = Wi[..., 1, t]
            w_list.append(w2)
        if S >= 3:
            w3 = Wi[..., 2, t]
            w_list.append(w3)
        if S >= 4:
            w4 = Wi[..., 3, t]
            w_list.append(w4)
        w_modes = [w1, w2, w3, w4]

        # robust per-frame scale
        s_flow = max(rob_scale(w) for w in [w_mean, w_real]) or 1.0
        s_modes = [
            rob_scale(w) if (w is not None and i < S) else 1.0
            for i, w in enumerate(w_modes)
        ]
        s_m1, s_m2, s_m3, s_m4 = s_modes

        # update images with scaled data (fixed [-1,0,1] norm)
        im_mean.set_data(w_mean / s_flow)
        im_real.set_data(w_real / s_flow)
        if S >= 1:
            im_m1.set_data(w1 / s_m1)
        if S >= 2:
            im_m2.set_data(w2 / s_m2)
        if S >= 3:
            im_m3.set_data(w3 / s_m3)
        if S >= 4:
            im_m4.set_data(w4 / s_m4)

        # update quivers
        q_mean.set_UVC(U[s, s, t], V[s, s, t])
        q_real.set_UVC(ur[s, s], vr[s, s])
        if S >= 1:
            q_m1.set_UVC(Ui[s, s, 0, t], Vi[s, s, 0, t])
        if S >= 2:
            q_m2.set_UVC(Ui[s, s, 1, t], Vi[s, s, 1, t])
        if S >= 3:
            q_m3.set_UVC(Ui[s, s, 2, t], Vi[s, s, 2, t])
        if S >= 4:
            q_m4.set_UVC(Ui[s, s, 3, t], Vi[s, s, 3, t])

        # move time marker on Var(Y_i) plot
        time_marker.set_xdata([Tt[t]])
        fig.suptitle(f"t = {Tt[t]:.3f}")
        return ()

    # ----- output path logic -----
    input_abs = os.path.abspath(args.input)
    in_dir = os.path.dirname(input_abs)
    base_name = os.path.splitext(os.path.basename(input_abs))[0]
    parent_dir = os.path.dirname(in_dir)

    if os.path.basename(in_dir) == "data":
        # .../parent/data/name.npz -> .../parent/movies/name_modes_variances.mp4
        out_dir = os.path.join(parent_dir, "movies")
    else:
        out_dir = args.out_dir

    os.makedirs(out_dir, exist_ok=True)
    output = os.path.join(out_dir, f"{base_name}_modes_variances.mp4")

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps, blit=False)
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=2000,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    update_func = lambda _i, _n: progress_bar.update(1)
    with tqdm(total=len(frames), desc="Saving video...") as progress_bar:
        ani.save(output, writer=writer, dpi=args.dpi, progress_callback=update_func)
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
