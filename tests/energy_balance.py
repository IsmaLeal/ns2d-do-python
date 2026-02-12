import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse, os, csv
import matplotlib.ticker as mtick
import pandas as pd

# ---------- helpers ---------- #

def leray_multipliers(KX, KY):
    K2 = KX*KX + KY*KY
    p1 = np.zeros_like(K2); p2 = np.zeros_like(K2); p3 = np.zeros_like(K2)
    mask = K2 != 0.0; invk2 = np.zeros_like(K2); invk2[mask] = 1.0/K2[mask]
    p1[mask] = (KY[mask]*KY[mask]) * invk2[mask]
    p2[mask] = -(KX[mask]*KY[mask]) * invk2[mask]
    p3[mask] = (KX[mask]*KX[mask]) * invk2[mask]
    return p1, p2, p3

def reconstruct_ou_hat(ou_a_row, half_k_pairs, KX, KY, Nx, Ny):
    # Normalize half_k_pairs to ndarray (nring,2)
    half_k_pairs = np.asarray(half_k_pairs, dtype=int).reshape(-1, 2)

    MC = ou_a_row.shape[0]
    Fx_hat = np.zeros((MC, Ny, Nx), dtype=np.complex128)
    Fy_hat = np.zeros((MC, Ny, Nx), dtype=np.complex128)

    for j, (ix_raw, iy_raw) in enumerate(half_k_pairs):
        a = ou_a_row[:, j]  # (MC,)
        ixp = ix_raw % Nx
        iyp = iy_raw % Ny
        ixn = (-ix_raw) % Nx
        iyn = (-iy_raw) % Ny

        kx = KX[iyp, ixp]
        ky = KY[iyp, ixp]
        k2 = kx * kx + ky * ky
        if k2 == 0.0:
            continue

        invk = 1.0 / np.sqrt(k2)
        ex, ey = -ky * invk, kx * invk
        Fxk = a * ex  # (MC,)
        Fyk = a * ey

        Fx_hat[:, iyp, ixp] += Fxk
        Fy_hat[:, iyp, ixp] += Fyk
        Fx_hat[:, iyn, ixn] += np.conj(Fxk)
        Fy_hat[:, iyn, ixn] += np.conj(Fyk)

    Fx_hat[:, 0, 0] = 0.0
    Fy_hat[:, 0, 0] = 0.0
    return Fx_hat, Fy_hat

def compute_metrics(u_save, v_save, ui_save, vi_save, w_save, wi_save, YY, Lx, Ly, nu, data, stride=1):
    Ny, Nx, T = u_save.shape
    dx = Lx / Nx; dy = Ly / Ny
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    Fx_hat_det, Fy_hat_det = data["Gxh"], data["Gyh"]

    idx = np.arange(0, T, stride, dtype=int)
    Ts = len(idx)

    # Mean quantities
    E_mean = np.empty(Ts); Z_mean = np.empty(Ts)
    # Modal quantities
    MC, S, _ = YY.shape
    G = np.zeros((Ts, S, S))  # Gram matrix of basis velocity fields
    G_omega = np.zeros((Ts, S, S))  # Gram matrix of vorticity modes
    C = np.zeros_like(G)  # Covariance matrix of stochastic coefficients
    E_modes = np.zeros(Ts); Z_modes = np.empty(Ts)
    for it, k in enumerate(tqdm(idx, desc="metrics being computed...")):
        # --- Mean quantities ---
        # energy
        u = u_save[...,k]; v = v_save[...,k]
        E_mean[it] = 0.5 * np.mean(u * u + v * v)
        # enstrophy
        omega = w_save[..., k]
        Z_mean[it] = 0.5 * np.mean(omega * omega)

        # --- Modal quantities ---
        S = ui_save.shape[2]
        if S > 0:
            # velocity modes for this timestep (shape (Ny, Nx, S))
            ui = ui_save[..., k]; vi = vi_save[..., k]
            # build Gram matrix of velocity modes for this timestep
            ui = ui.reshape(-1, S); vi = vi.reshape(-1, S)  # flatten each mode's values into a Ny*Nx column
            G[it] = (ui.T @ ui + vi.T @ vi) / (Nx * Ny)

            # build Gram matrix of vorticity modes for this timestep
            wi = wi_save[..., k]
            Wi = wi.reshape(-1, S)  # flatten each mode's values into a Ny*Nx column
            G_omega[it] = (Wi.T @ Wi) / (Nx * Ny)

            # build covariance matrix for this timestep
            Y = YY[..., k]
            Ck = (Y.T @ Y) / Y.shape[0]
            C[it] = Ck
            # get modal energy at this timestep
            E_modes[it] = 0.5 * np.trace(Ck @ G[it])
            # get modal enstrophy at this timestep
            Z_modes[it] = 0.5 * np.trace(C[it] @ G_omega[it])
    E_tot = E_mean + E_modes
    Z_tot = Z_mean + Z_modes
    eps = 2 * nu * Z_tot    # total dissipation

    return (E_tot, Z_tot, eps,
            Fx_hat_det, Fy_hat_det, KX, KY, idx)

# ---------- main ---------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--data_available", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    if args.data_available == 0:
        data = np.load(args.input, allow_pickle=True)
        u_save = data["U_save"]; v_save = data["V_save"]
        ui_save = data["Ui_save"]; vi_save = data["Vi_save"]
        w_save = data["W_save"]; wi_save = data["Wi_save"]
        YY = data["YY_save"]
        MC = data["MC"]
        Lx, Ly = float(data["Lx"]), float(data["Ly"])
        nu = 1 / float(data["Re"])
        forcing = str(data["Forcing"])
        stochastic = str(data["stochastic"]) if "stochastic" in data else "none"

        print("computing metrics...")
        (E, Z, eps,
         Fx_hat_det, Fy_hat_det, KX, KY, idx) = compute_metrics(u_save, v_save, ui_save, vi_save, w_save, wi_save, YY, Lx, Ly, nu, data, stride=args.stride)
        fx_det = np.fft.ifft2(Fx_hat_det).real  # (Ny,Nx)
        fy_det = np.fft.ifft2(Fy_hat_det).real
        print("metrics computed!")
        Ny, Nx, T = u_save.shape
        Ts = len(idx)
        # Build OU power time series + (for checks) spectral OU forcing each step
        P_tot = np.empty(Ts)
        P_det = np.empty(Ts)
        P_mc = np.empty(Ts)
        E_mc = np.empty(Ts)
        eps_mc = np.empty(Ts)
        have_ou = not any(x is None for x in data["ou_amplitudes"])
        hk = None
        if have_ou:
            hk_raw = data["half_k_pairs"]
            hk = np.array(hk_raw, dtype=int).reshape(-1, 2) if np.size(hk_raw) else np.zeros((0,2), dtype=int)
            ou_amplitudes = data["ou_amplitudes"]

        # time base
        t_full = np.array(data["T_save"], dtype=float)
        t = t_full[idx]

        # window (last third)
        w0 = int(len(E) * 0.55); W = slice(w0, None)

        # Parseval + solenoidality diagnostics (max over W)
        max_rel_parseval = 0.0
        max_rel_divfree  = 0.0

        for it, n in enumerate(tqdm(idx, desc="energies now")):
            # --- Mean fields at this saved time ---
            u_mean = u_save[..., n]
            v_mean = v_save[..., n]
            w_mean = w_save[..., n]

            # --- Build OU forcing (spectral) per realisation, then total forcing (spectral) ---
            if have_ou:
                # data["ou_amplitudes"][n] now has shape (MC, nring)
                Fx_hat_ou_s, Fy_hat_ou_s = reconstruct_ou_hat(
                    ou_amplitudes[n], hk, KX, KY, Nx, Ny
                )  # → (MC, Ny, Nx)

            else:
                Fx_hat_ou_s = np.zeros((MC, Ny, Nx), dtype=np.complex128)
                Fy_hat_ou_s = np.zeros_like(Fx_hat_ou_s)

            # spectral forcing in physical space
            fx_stoch_s = np.fft.ifft2(Fx_hat_ou_s, axes=(1, 2)).real
            fy_stoch_s = np.fft.ifft2(Fy_hat_ou_s, axes=(1, 2)).real

            # Back to physical space per realisation
            fx_tot_s = fx_stoch_s + fx_det[None, ...]
            fy_tot_s = fy_stoch_s + fy_det[None, ...]

            # Reconstruct total velocity for all realisations at once
            MC, S, _ = YY.shape
            Ui_n = ui_save[..., n]  # (Ny,Nx,S)
            Vi_n = vi_save[..., n]  # (Ny,Nx,S)
            wi_n = wi_save[..., n]
            Yr = YY[:, :, n]        # (MC,S)

            u_rec_all = u_mean[..., None] + np.tensordot(Ui_n, Yr.T, axes=(2, 0))
            v_rec_all = v_mean[..., None] + np.tensordot(Vi_n, Yr.T, axes=(2, 0))
            w_rec_all = w_mean[..., None] + np.tensordot(wi_n, Yr.T, axes=(2, 0))
            eps_mc[it] = 2 * nu * (0.5 * np.mean(w_rec_all**2, axis=(0, 1))).mean()

            # --- Injected power via decomposition: <u_mean, f_det> + Σ_i <u_i, E[f_stoch * Y_i]> ---
            # 1) deterministic contribution (same for all realisations)
            P_det_scalar = np.mean(u_mean * fx_det + v_mean * fy_det)  # <ū, f_det>

            # 2) build E[f_stoch * Y_i] in physical space, per component (Ny, Nx, S)
            # Yr: (MC, S)
            EfYx = (fx_stoch_s[:, :, :, None] * Yr[:, None, None, :]).mean(axis=0)  # (Ny, Nx, S)
            EfYy = (fy_stoch_s[:, :, :, None] * Yr[:, None, None, :]).mean(axis=0)  # (Ny, Nx, S)

            # 3) cross-term Σ_i < (u_i, v_i), (E[f_x Y_i], E[f_y Y_i]) >
            cross = np.mean(Ui_n * EfYx + Vi_n * EfYy, axis=(0, 1)).sum()

            # 4) terms that tend to 0 as $MC\to\infty$
            Efxstoch, Efystoch = np.mean(fx_stoch_s, axis=0), np.mean(fy_stoch_s, axis=0)
            term1 = np.mean(u_mean * Efxstoch + v_mean * Efystoch)
            EY = np.mean(Yr, axis=0)
            term2 = 0.0
            for i in range(Ui_n.shape[2]):
                term2 += EY[i] * np.mean(Ui_n[..., i] * fx_det + Vi_n[..., i] * fy_det)

            P_det[it] = P_det_scalar
            P_tot[it] = P_det_scalar + cross + term1 + term2

            fx_tot_all = np.moveaxis(fx_tot_s, 0, 2)
            fy_tot_all = np.moveaxis(fy_tot_s, 0, 2)

            E_mc[it] = (0.5 * np.mean(u_rec_all**2 + v_rec_all**2, axis=(0,1))).mean()
            P_mc[it] = np.mean(u_rec_all * fx_tot_all + v_rec_all * fy_tot_all)

        # compute residual from empirical dE/dt
        dt_phys = np.mean(np.diff(t))
        dE_dt = np.gradient(E, dt_phys)
        R = np.abs(dE_dt - P_tot + eps)

        # empirical MC residual
        dE_dt_mc = np.gradient(E_mc, dt_phys)
        R_mc = dE_dt_mc - P_mc + eps_mc

        print("⟨residual⟩ over window =", R[W].mean(), " rms =", R[W].std())

        print("⟨dE/dt⟩ vs ⟨P_tot⟩-⟨ε⟩:",
              dE_dt[W].mean(), (P_tot[W].mean() - eps[W].mean()))

        # Budget + stationarity
        r = abs(dE_dt[W].mean()) / (eps[W].mean() if eps[W].mean()!=0 else 1.0)
        budget_residual = (dE_dt[W].mean() - P_tot[W].mean() + eps[W].mean())

        print(rf"[checks] r = |<dE/dt>|/<ε> over window = {r:.3f}")
        print(f"[checks] budget residual <P_tot>-<ε>-<dE/dt> over window = {budget_residual:.3e}")
        print(f"[checks] Parseval max rel error over window = {max_rel_parseval:.3e}")
        print(f"[checks] Forcing solenoidality max rel error over window = {max_rel_divfree:.3e}")

        # ---------------- plots ---------------- #
        plt.style.use("seaborn-v0_8")
        plt.rcParams.update({
            "font.size": 17,
            "legend.fontsize": 15,
            "axes.labelsize": 13,
            "font.family": "serif"
        })
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))

        # TOP: split scales
        axE = axs  # left y: energy
        axP = axE.twinx()  # right y: power terms

        # left y-axis
        line_E, = axE.plot(t, E, color="tab:blue", lw=2.0, label=r"$E_{tot}(t)$")

        # right y-axis
        line_eps, = axP.plot(t, eps, color="tab:red", lw=1.6, label=r"$\varepsilon(t)$")
        line_Pt, = axP.plot(t, P_tot, color="tab:olive", lw=1.6, label=r"$P_{tot}(t)$")
        line_Pd, = axP.plot(t, P_det, color="tab:green", ls="--", lw=1.2, label=r"$P_{det}(t)$")
        line_R, = axP.plot(t, dE_dt - P_tot + eps, color="tab:brown", ls="-.", lw=1.2,
                           label=r"dE/dt - $P_{tot}+\varepsilon$")

        # # averaging window
        # axE.axvspan(t[w0], t[-1], color="k", alpha=0.05)

        # labels/titles
        axE.set_xlabel("Time")
        axE.set_ylabel(r"kinetic energy $E_{tot}$", color="tab:blue")
        axP.set_ylabel("Power terms  (P, ε, residual)")

        # tidy ticks + grid
        axE.yaxis.set_major_locator(mtick.MaxNLocator(6))
        axP.yaxis.set_major_locator(mtick.MaxNLocator(6))
        axE.grid(True, which="both", alpha=0.6)
        axP.grid(False)
        axE.tick_params(axis="y", colors="C0")

        # single legend outside (right)
        lines = [line_E, line_eps, line_Pt, line_Pd, line_R]
        labels = [L.get_label() for L in lines]
        legE = axs.legend(lines, labels, loc="upper right", bbox_to_anchor=(1, 0.6), frameon=True, fontsize=11)
        legE.get_frame().set_edgecolor("black")

        # # BOTTOM: enstrophy on semilog-y
        # axZ = axs[1]
        # axZ.semilogy(t, Z, color="C4", lw=1.8, label="Z(t)")
        # axZ.set_xlabel("Time")
        # axZ.set_ylabel("Enstrophy  Z(t)")
        # # log ticks: decades + labeled minor ticks at 2..9
        # axZ.yaxis.set_major_locator(mtick.LogLocator(base=10.0))
        # axZ.yaxis.set_minor_locator(mtick.LogLocator(base=10.0, subs=np.arange(2, 10, 2) * 0.1))
        # axZ.yaxis.set_major_formatter(mtick.LogFormatter(base=10.0, labelOnlyBase=True))
        # axZ.yaxis.set_minor_formatter(mtick.LogFormatter(base=10.0, labelOnlyBase=False))
        # axZ.yaxis.get_minor_formatter().minor_thresholds = (np.inf, np.inf)
        # axZ.tick_params(axis="y", which="minor", labelsize=9)
        # axZ.grid(True, which="both", alpha=0.6)
        # legZ = axZ.legend(loc="upper right", bbox_to_anchor=(1, 1), frameon=True, fontsize=12)
        # legZ.get_frame().set_edgecolor("black")
        # axs[1].set_title("Enstrophy evolution")

        # layout: leave right margin for legend
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        # save figure + CSV (unchanged)
        base_dir = os.path.dirname(args.input)  # e.g. F-none/data
        parent_dir = os.path.dirname(base_dir)  # e.g. F-none
        plots_dir = os.path.join(parent_dir, "plots")  # e.g. F-none/plots
        os.makedirs(plots_dir, exist_ok=True)

        # save plot in F-none/plots/
        basename = os.path.splitext(os.path.basename(args.input))[0]  # fields
        outpath = os.path.join(plots_dir, basename + f"_analysis.png")
        fig.savefig(outpath, dpi=150)
        print("Saved", outpath)

        outpath2 = os.path.splitext(args.input)[0] + "_energies.csv"
        arr = np.vstack([t, E, dE_dt, P_det, P_tot, eps, R, E_mc, P_mc, eps_mc]).T
        header = "t,E,dE_dt,P_det,P_tot,eps,residual,E_mc,P_mc,eps_mc"
        np.savetxt(outpath2, arr, delimiter=",", header=header)

        summary_path = os.path.splitext(args.input)[0] + "_summary_checks.csv"
        row = [
            args.input,
            forcing,
            stochastic,
            dt_phys,
            MC,
            nu,
            r,
            budget_residual,
            max_rel_parseval,
            max_rel_divfree,
            R[W].mean(),
            R[W].std(),
        ]
        summary_header = ["file", "forcing", "stochastic", "dt", "MC", "nu",
                          "r", "budget_residual", "max_rel_parseval",
                          "max_rel_divfree", "R_mean", "R_rms"]

        if not os.path.exists(summary_path):
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(summary_header)
                writer.writerow(row)
        else:
            with open(summary_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
    else:
        energies = pd.read_csv(args.input)
        t = energies["# t"].values
        E = energies["E"].values
        dE_dt = energies["dE_dt"].values
        P_det = energies["P_det"].values
        P_tot = energies["P_tot"].values
        eps = energies["eps"].values
        w0 = int(len(E) * 0.35)

        # save figure + CSV (unchanged)
        base_dir = os.path.dirname(args.input)  # e.g. F-none/data
        summary = os.path.join(base_dir, "fields_ouamp-1.0_summary_checks.csv")
        summary_checks = pd.read_csv(summary)
        nu = summary_checks["nu"].values[0]
        Z = eps / (2*nu)

        # ---------------- plots ---------------- #
        plt.style.use("seaborn-v0_8")
        plt.rcParams.update({
            "font.size": 17,
            "legend.fontsize": 15,
            "axes.labelsize": 13,
            "font.family": "serif"
        })
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))

        axE = axs  # left y: energy
        axP = axE.twinx()  # right y: power terms

        # left y — no clipping of E
        line_E, = axE.plot(t, E, color="tab:blue", lw=2.0, label=r"$E_{tot}(t)$")

        # right y — powers + residual
        line_eps, = axP.plot(t, eps, color="tab:red", lw=1.6, label=r"$\varepsilon(t)$")
        line_Pt, = axP.plot(t, P_tot, color="tab:olive", lw=1.6, label=r"$P_{tot}(t)$")
        line_Pd, = axP.plot(t, P_det, color="tab:green", ls="--", lw=1.2, label=r"$P_{det}(t)$")
        line_R, = axP.plot(t, dE_dt - P_tot + eps, color="tab:brown", ls="-.", lw=1.2,
                           label=r"dE/dt - $P_{tot}+\varepsilon$")

        # averaging window
        # axE.axvspan(t[w0], t[-1], color="k", alpha=0.1)

        # labels/titles
        axE.set_xlabel("Time")
        axE.set_ylabel(r"Kinetic energy $E_{tot}$", color="tab:blue")
        axP.set_ylabel("Power terms  (P, ε, residual)")

        # tidy ticks + grid
        axE.yaxis.set_major_locator(mtick.MaxNLocator(6))
        axP.yaxis.set_major_locator(mtick.MaxNLocator(6))
        axE.grid(True, which="both", alpha=0.6)
        axP.grid(False)
        axE.tick_params(axis="y", colors="C0")

        # single legend outside (right)
        lines = [line_E, line_eps, line_Pt, line_Pd, line_R]
        labels = [L.get_label() for L in lines]
        legE = axs.legend(lines, labels, loc="upper right", bbox_to_anchor=(1, 0.9), frameon=True, fontsize=11)
        legE.get_frame().set_edgecolor("black")

        # # BOTTOM: enstrophy on semilog-y
        # axZ = axs[1]
        # axZ.semilogy(t, Z, color="C4", lw=1.8, label="Z(t)")
        # axZ.set_xlabel("Time")
        # axZ.set_ylabel("Enstrophy  Z(t)")
        # # log ticks: decades + labeled minor ticks at 2..9
        # axZ.yaxis.set_major_locator(mtick.LogLocator(base=10.0))
        # axZ.yaxis.set_minor_locator(mtick.LogLocator(base=10.0, subs=np.arange(2, 10, 2) * 0.1))
        # axZ.yaxis.set_major_formatter(mtick.LogFormatter(base=10.0, labelOnlyBase=True))
        # axZ.yaxis.set_minor_formatter(mtick.LogFormatter(base=10.0, labelOnlyBase=False))
        # axZ.yaxis.get_minor_formatter().minor_thresholds = (np.inf, np.inf)
        # axZ.tick_params(axis="y", which="minor", labelsize=9)
        # axZ.grid(True, which="both", alpha=0.6)
        # legZ = axZ.legend(loc="upper right", bbox_to_anchor=(1, 1), frameon=True, fontsize=12)
        # legZ.get_frame().set_edgecolor("black")
        # axZ.set_title("Enstrophy evolution")

        # layout: leave right margin for legend
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        parent_dir = os.path.dirname(base_dir)  # e.g. F-none
        plots_dir = os.path.join(parent_dir, "plots")  # e.g. F-none/plots
        os.makedirs(plots_dir, exist_ok=True)

        basename = os.path.splitext(os.path.basename(args.input))[0].removesuffix("_energies")
        outpath = os.path.join(plots_dir, basename + f"_analysis.png")
        fig.savefig(outpath, dpi=150)
        print("Saved", outpath)


if __name__ == "__main__":
    main()
