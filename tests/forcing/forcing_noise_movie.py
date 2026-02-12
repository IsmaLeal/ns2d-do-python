from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import os

# -------------------- utilities --------------------
def leray_multipliers(KX, KY):
    K2 = KX*KX + KY*KY
    p1 = np.zeros_like(K2); p2 = np.zeros_like(K2); p3 = np.zeros_like(K2)
    mask = K2 != 0.0
    invk2 = np.zeros_like(K2); invk2[mask] = 1.0/K2[mask]
    p1[mask] = (KY[mask]*KY[mask]) * invk2[mask]
    p2[mask] = -(KX[mask]*KY[mask]) * invk2[mask]
    p3[mask] = (KX[mask]*KX[mask]) * invk2[mask]
    return p1, p2, p3

def fft_shift(Fxh, Fyh, f):
    Fx = np.fft.fftshift(Fxh[..., f])
    Fy = np.fft.fftshift(Fyh[..., f])
    return np.log1p(np.hypot(np.abs(Fx), np.abs(Fy)))

def reconstruct_ou_hat(data, n, half_k_pairs, KX, KY, Nx, Ny, mc):
    a_block = data["ou_amplitudes"][n]  # (nring,) or (MC,nring)
    ou_a = a_block[mc] if a_block.ndim == 2 else a_block

    Fx_hat = np.zeros((Ny, Nx), dtype=np.complex128)
    Fy_hat = np.zeros((Ny, Nx), dtype=np.complex128)

    for a, (ix_raw, iy_raw) in zip(ou_a, half_k_pairs):
        ixp = ix_raw % Nx;  iyp = iy_raw % Ny
        ixn = (-ix_raw) % Nx; iyn = (-iy_raw) % Ny
        kx = KX[iyp, ixp]; ky = KY[iyp, ixp]
        k2 = kx*kx + ky*ky
        if k2 == 0.0:
            continue
        invk = 1.0/np.sqrt(k2)
        ex, ey = -ky*invk, kx*invk
        Fxk, Fyk = a*ex, a*ey
        Fx_hat[iyp, ixp] += Fxk; Fy_hat[iyp, ixp] += Fyk
        Fx_hat[iyn, ixn] += np.conj(Fxk); Fy_hat[iyn, ixn] += np.conj(Fyk)

    Fx_hat[0,0] = 0.0; Fy_hat[0,0] = 0.0
    return Fx_hat, Fy_hat

def reconstruct_f_det(data):
    Ny, Nx, _ = data["U_save"].shape
    dx = data["Lx"]/data["Nx"]; dy = data["Ly"]/data["Ny"]
    x = np.arange(0, data["Lx"], dx)
    y = np.arange(0, data["Ly"], dy)
    X, Y = np.meshgrid(x, y)

    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)

    p1, p2, p3 = leray_multipliers(KX, KY)
    Fx_hat = p1*data["Gxh"] + p2*data["Gyh"]
    Fy_hat = p2*data["Gxh"] + p3*data["Gyh"]
    return Fx_hat, Fy_hat, KX, KY, Nx, Ny, X, Y

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mc", type=int)
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    Nt = int(data["U_save"].shape[2])
    t = np.array(data["T_save"], dtype=float)

    Fx_hat_det, Fy_hat_det, KX, KY, Nx, Ny, X, Y = reconstruct_f_det(data)

    have_ou = ("ou_amplitudes" in data and "half_k_pairs" in data and data["ou_amplitudes"].size)
    half_k_pairs = np.array(data["half_k_pairs"], int).reshape(-1,2) if have_ou else np.zeros((0,2),int)
    mc = args.mc if args.mc is not None else 0

    # ---- build time series ----
    Fx_hat_OU = np.zeros((Ny, Nx, Nt), complex)
    Fy_hat_OU = np.zeros((Ny, Nx, Nt), complex)
    Fx_OU = np.zeros((Ny, Nx, Nt))
    Fy_OU = np.zeros((Ny, Nx, Nt))

    Fx_det = np.fft.ifft2(Fx_hat_det).real[...,None]
    Fy_det = np.fft.ifft2(Fy_hat_det).real[...,None]

    for n in tqdm(range(Nt)):
        if have_ou and half_k_pairs.size:
            Fxh, Fyh = reconstruct_ou_hat(data, n, half_k_pairs, KX, KY, Nx, Ny, mc)
        else:
            Fxh = np.zeros_like(Fx_hat_det); Fyh = np.zeros_like(Fy_hat_det)
        Fx_hat_OU[...,n] = Fxh; Fy_hat_OU[...,n] = Fyh
        Fx_OU[...,n] = np.fft.ifft2(Fxh).real
        Fy_OU[...,n] = np.fft.ifft2(Fyh).real

    Fx_tot = Fx_OU + Fx_det
    Fy_tot = Fy_OU + Fy_det

    # ---- animation params ----
    frames = list(range(0, Nt, 10)) or [0]
    stride_phys = 4; stride_spec = 1
    Ys = slice(None,None,stride_phys); Xs = slice(None,None,stride_phys)
    Yks = slice(None,None,stride_spec); Xks = slice(None,None,stride_spec)
    Xs2, Ys2 = X[Ys,Xs], Y[Ys,Xs]
    KXc = np.fft.fftshift(KX)
    KYc = np.fft.fftshift(KY)
    KXs, KYs = KXc[Yks,Xks], KYc[Yks,Xks]

    def max_phys(FxA, FyA):
        return max(np.max(np.hypot(FxA[...,f], FyA[...,f])) for f in frames)

    def max_spec(FxH, FyH):
        return max(np.max(np.hypot(np.abs(FxH[...,f]), np.abs(FyH[...,f]))) for f in frames)

    M_phys = max(np.max(np.hypot(Fx_det[...,0], Fy_det[...,0])),
                 max_phys(Fx_OU, Fy_OU),
                 max_phys(Fx_tot, Fy_tot))
    M_spec = max_spec(Fx_hat_OU, Fy_hat_OU)

    L_phys = 0.12*min(X.max()-X.min(), Y.max()-Y.min())
    L_spec = 0.12*min(KX.max()-KX.min(), KY.max()-KY.min())
    scale_phys = M_phys/L_phys if M_phys>0 else 1.0
    scale_spec = M_spec/L_spec if M_spec>0 else 1.0

    # ---- shared physical background scale ----
    phys_vmax = M_phys if M_phys>0 else 1.0
    cmap_phys = "viridis"

    def phys_mag(FxA, FyA, f): return np.hypot(FxA[...,f], FyA[...,f])
    def spec_mag(FxH, FyH, f): return np.log1p(np.hypot(np.abs(FxH[...,f]), np.abs(FyH[...,f])))

    # ---- figure ----
    fig, axs = plt.subplots(2,2, figsize=(12,5), constrained_layout=True)
    axdet, axp = axs[0,0], axs[0,1]
    axtot, axk = axs[1,0], axs[1,1]

    extent_phys = [X.min(), X.max(), Y.min(), Y.max()]
    extent_spec = [KXc.min(), KXc.max(), KYc.min(), KYc.max()]
    i0 = frames[0]

    Im_det = axdet.imshow(phys_mag(Fx_det, Fy_det, 0), origin="lower",
                          extent=extent_phys, vmin=0, vmax=phys_vmax,
                          cmap=cmap_phys, aspect="equal")
    Im_ou  = axp.imshow(phys_mag(Fx_OU, Fy_OU, i0), origin="lower",
                        extent=extent_phys, vmin=0, vmax=phys_vmax,
                        cmap=cmap_phys, aspect="equal")
    Im_tot = axtot.imshow(phys_mag(Fx_tot, Fy_tot, i0), origin="lower",
                          extent=extent_phys, vmin=0, vmax=phys_vmax,
                          cmap=cmap_phys, aspect="equal")

    spec_vmax = max(np.max(fft_shift(Fx_hat_OU, Fy_hat_OU, f)) for f in frames) or 1.0
    Im_k = axk.imshow(fft_shift(Fx_hat_OU, Fy_hat_OU, i0), origin="lower",
                      extent=extent_spec, vmin=0, vmax=spec_vmax, aspect="equal")

    cbar = fig.colorbar(Im_tot, ax=[axdet, axp, axtot], fraction=0.046, pad=0.04)
    cbar.set_label(r"$|f(x,y)|$")
    fig.colorbar(Im_k, ax=axk, fraction=0.046, pad=0.04,
                 label=r"$\log(1+|\hat f|)$")

    Qdet = axdet.quiver(Xs2, Ys2, Fx_det[Ys,Xs,0], Fy_det[Ys,Xs,0],
                        angles="xy", scale_units="xy", scale=scale_phys,
                        units="inches", width=0.004, pivot="mid")
    Qp   = axp.quiver(Xs2, Ys2, Fx_OU[Ys,Xs,i0], Fy_OU[Ys,Xs,i0],
                      angles="xy", scale_units="xy", scale=scale_phys,
                      units="inches", width=0.004, pivot="mid")
    Qtot = axtot.quiver(Xs2, Ys2, Fx_tot[Ys,Xs,i0], Fy_tot[Ys,Xs,i0],
                        angles="xy", scale_units="xy", scale=scale_phys,
                        units="inches", width=0.004, pivot="mid")
    Fxh0 = np.fft.fftshift(Fx_hat_OU[..., i0])
    Fyh0 = np.fft.fftshift(Fy_hat_OU[..., i0])
    Qk   = axk.quiver(KXs, KYs, Fxh0[Yks,Xks].real, Fyh0[Yks,Xks].real,
                      angles="xy", scale_units="xy", scale=scale_spec,
                      units="inches", width=0.004, pivot="mid")

    axdet.set_title("Deterministic forcing")
    axp.set_title("Forcing noise")
    axtot.set_title("Total forcing")
    axk.set_title("Forcing (spectral)")

    def update(frame):
        Im_ou.set_data(phys_mag(Fx_OU, Fy_OU, frame))
        Im_tot.set_data(phys_mag(Fx_tot, Fy_tot, frame))
        Qp.set_UVC(Fx_OU[Ys,Xs,frame], Fy_OU[Ys,Xs,frame])
        Qtot.set_UVC(Fx_tot[Ys,Xs,frame], Fy_tot[Ys,Xs,frame])
        Fxhf = np.fft.fftshift(Fx_hat_OU[..., frame])
        Fyhf = np.fft.fftshift(Fy_hat_OU[..., frame])
        Im_k.set_data(np.log1p(np.hypot(np.abs(Fxhf), np.abs(Fyhf))))
        Qk.set_UVC(Fxhf[Yks, Xks].real, Fyhf[Yks, Xks].real)
        fig.suptitle(f"Time = {t[frame]:.3f}" if t.size else f"Frame {frame}")
        return Im_ou, Im_tot, Im_k, Qp, Qtot, Qk

    writer = FFMpegWriter(fps=max(1, int(1000*(len(frames))/(t[-1]*len(frames)))/2 if t.size else 20),
                          codec="libx264", bitrate=2000,
                          extra_args=["-pix_fmt","yuv420p"])

    outdir = os.path.join(os.path.dirname(os.path.dirname(args.input)), "movies")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, os.path.splitext(os.path.basename(args.input))[0] + "_forcing.mp4")

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=False, cache_frame_data=False)
    ani.save(outpath, writer=writer, dpi=150)
    print("saved:", outpath)

if __name__ == "__main__":
    main()
