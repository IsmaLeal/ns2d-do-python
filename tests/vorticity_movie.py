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
    W = Z["W_save"]         # (Ny, Nx, T)
    Wi = Z["Wi_save"]       # (Ny, Nx, S, T)
    Tt = Z["T_save"]        # (T,)

    Lx, Ly = float(Z["Lx"]), float(Z["Ly"])
    Ny, Nx, T = U.shape
    S = Ui.shape[2]
    MC = YY.shape[0]
    r = np.random.randint(0, MC) if args.r is None else int(np.clip(args.r, 0, MC-1))

    # grid
    dx = Lx/Nx; dy = Ly/Ny
    x = np.arange(0, Lx, dx)[:Nx]
    y = np.arange(0, Ly, dy)[:Ny]
    extent = [x.min(), x.max(), y.min(), y.max()]

    # --- quiver settings ---
    X, Y = np.meshgrid(x, y)
    s = slice(None, None, args.stride)
    Xs, Ys = X[s, s], Y[s, s]

    # global arrow scale from max speed over time (mean + one realisation)
    urec = U + np.einsum('xyst,st->xyt', Ui, YY[r])
    vrec = V + np.einsum('xyst,st->xyt', Vi, YY[r])
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
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])

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

    w_m10 = Wi[..., 0, t0] if S >= 1 else None
    w_m20 = Wi[..., 1, t0] if S >= 2 else None
    w_m30 = Wi[..., 2, t0] if S >= 3 else None
    w_m40 = Wi[..., 3, t0] if S >= 4 else None

    # helper: one panel
    def add_panel(ax, w_data, u_data, v_data, title, cbar_label):
        im = ax.imshow(w_data, origin="lower", extent=extent, cmap=cmap, norm=norm01, interpolation="bilinear")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        quiv = ax.quiver(
            Xs, Ys, u_data[s, s], v_data[s, s],
            color="k", angles="xy", scale_units="xy",
            scale=scale, units="inches", width=args.arrwidth, pivot="mid"
        )
        return im, cbar, quiv

    # initial robust scale s(t0) across visible panels
    s_flow0 = max(rob_scale(w_mean0), rob_scale(w_real0)) or 1.0
    s_modes0 = max((rob_scale(w) for w in [w_m10, w_m20, w_m30, w_m40] if w is not None), default=1.0)

    # draw initial scaled images
    im_mean, cb_mean, q_mean = add_panel(ax_mean, w_mean0/s_flow0, U[..., t0], V[..., t0], r"Mean vorticity $\bar{\omega}$", r"$\omega/s(t)$")
    im_real, cb_real, q_real = add_panel(ax_real, w_real0/s_flow0, u0, v0, f"DO realisation r={r}: $\\omega^{{(r)}}$", r"$\omega/s(t)$")
    cb_mean.set_label(r'$\omega/s_{\rm flow}(t)$')
    cb_real.set_label(r'$\omega/s_{\rm flow}(t)$')
    if S >= 1:
        im_m1, cb_m1, q_m1 = add_panel(ax_m1, (YY[r, 0, t0] * w_m10)/s_modes0, YY[r, 0, t0] * Ui[..., 0, t0], YY[r, 0, t0] * Vi[..., 0, t0], r"Mode 1 vorticity $Y_1\omega_1$", r"$Y_1\omega_1/s(t)$")
        cb_m1.set_label(r'$Y_1\omega_1/s_{\rm modes}(t)$')
    if S >= 2:
        im_m2, cb_m2, q_m2 = add_panel(ax_m2,   ((YY[r, 1, t0] * w_m20)/s_modes0) if S>=2 else np.zeros_like(w_mean0), YY[r, 1, t0] * Ui[..., 1, t0], YY[r, 1, t0] * Vi[..., 1, t0], r"Mode 2 vorticity $Y_2\omega_2$", r"$Y_2\omega_2/s(t)$")
        cb_m2.set_label(r'$Y_2\omega_2/s_{\rm modes}(t)$')
    if S >= 3:
        im_m3, cb_m3, q_m3 = add_panel(ax_m3, ((YY[r, 2, t0] * w_m30)/s_modes0) if S>=3 else np.zeros_like(w_mean0), YY[r, 2, t0] * Ui[..., 2, t0], YY[r, 2, t0] * Vi[..., 2, t0], r"Mode 3 vorticity $Y_3\omega_3$", r"$Y_3\omega_3/s(t)$")
        cb_m3.set_label(r'$Y_3\omega_3/s_{\rm modes}(t)$')
    if S >= 4:
        im_m4, cb_m4, q_m4 = add_panel(ax_m4, ((YY[r, 3, t0] * w_m40)/s_modes0) if S>=4 else np.zeros_like(w_mean0), YY[r, 3, t0] * Ui[..., 3, t0], YY[r, 3, t0] * Vi[..., 3, t0], r"Mode 4 vorticity $Y_4\omega_4$", r"$Y_4\omega_4/s(t)$")
        cb_m4.set_label(r'$Y_4\omega_4/s_{\rm modes}(t)$')


    # bottom plot: Yi(t)
    lines = []
    for i in range(min(4, S)):
        (ln,) = ax_Y.plot(Tt, YY[r, i, :], label=f"$Y_{i+1}$")
        lines.append(ln)
    ax_Y.set_xlim(Tt[0], Tt[-1]); ax_Y.set_xlabel("t"); ax_Y.set_ylabel("$Y_i$")
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
        if S >= 1: w1 = YY[r, 0, t] * Wi[..., 0, t]; w_list.append(w1)
        if S >= 2: w2 = YY[r, 1, t] * Wi[..., 1, t]; w_list.append(w2)
        if S >= 3: w3 = YY[r, 2, t] * Wi[..., 2, t]; w_list.append(w3)
        if S >= 4: w4 = YY[r, 3, t] * Wi[..., 3, t]; w_list.append(w4)

        # robust per-frame scale
        s_flow = max(rob_scale(w) for w in [w_mean, w_real]) or 1.0
        s_modes = max((rob_scale(w) for w in [w1, w2, w3, w4] if w is not None), default=1.0)

        # update images with scaled data (fixed [-1,0,1] norm)
        im_mean.set_data(w_mean / s_flow)
        im_real.set_data(w_real / s_flow)
        if S >= 1: im_m1.set_data(w1 / s_modes)
        if S >= 2: im_m2.set_data(w2 / s_modes)
        if S >= 3: im_m3.set_data(w3 / s_modes)
        if S >= 4: im_m4.set_data(w4 / s_modes)

        # update quivers
        q_mean.set_UVC(U[s, s, t], V[s, s, t])
        q_real.set_UVC(ur[s, s],   vr[s, s])
        if S >= 1: q_m1.set_UVC(YY[r, 0, t] * Ui[s, s, 0, t], YY[r, 0, t] * Vi[s, s, 0, t])
        if S >= 2: q_m2.set_UVC(YY[r, 1, t] * Ui[s, s, 1, t], YY[r, 1, t] * Vi[s, s, 1, t])
        if S >= 3: q_m3.set_UVC(YY[r, 2, t] * Ui[s, s, 2, t], YY[r, 2, t] * Vi[s, s, 2, t])
        if S >= 4: q_m4.set_UVC(YY[r, 3, t] * Ui[s, s, 3, t], YY[r, 3, t] * Vi[s, s, 3, t])

        time_marker.set_xdata([Tt[t]])
        fig.suptitle(f"t = {Tt[t]:.3f}")
        return ()

    # save
    data_dir = os.path.dirname(args.input)
    base_dir = os.path.dirname(data_dir)
    movies_dir = os.path.join(base_dir, "movies")
    os.makedirs(movies_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output = os.path.join(movies_dir, f"{base_name}_vorticity_r{r}.mp4")

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1000/args.fps, blit=False)
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=2000, extra_args=["-pix_fmt", "yuv420p"])
    update_func = lambda _i, _n: progress_bar.update(1)
    with tqdm(total=len(frames), desc="Saving video...") as progress_bar:
        ani.save(output, writer=writer, dpi=args.dpi, progress_callback=update_func)
    print(f"Saved: {output}")

if __name__ == "__main__":
    main()
