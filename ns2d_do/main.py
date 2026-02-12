from __future__ import annotations
import argparse
import math
import os
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DO 2D NS: main driver")
    # domain/grid
    p.add_argument("--Lx", type=float, default=2.0 * math.pi, help="domain length in x")
    p.add_argument("--Ly", type=float, default=2.0 * math.pi, help="domain length in y")
    p.add_argument("--Nx", type=int, default=64, help="grid points in x (even)")
    p.add_argument("--Ny", type=int, default=64, help="grid points in y (even)")
    # time
    p.add_argument("--Tf", type=float, default=20, help="final time")
    p.add_argument("--dt", type=float, default=1e-3, help="time step")
    # DO sizes
    p.add_argument("--S", type=int, default=10, help="number of DO modes")
    p.add_argument("--MC", type=int, default=200, help="number of coefficient samples1 to keep")
    # physics
    p.add_argument("--Re", type=float, default=40.0, help="Reynolds number")
    p.add_argument("--forcing", type=str, default="none", choices=["none", "kick", "kolmogorov"], help="body force type")
    p.add_argument("--f_amp", type=float, default=1.0, help="deterministic forcing amplitude")
    p.add_argument("--forcing_stochastic", type=str, default="spectral_ou", choices=["none", "spectral_ou"], help="stochastic forcing type")
    p.add_argument("--ou_tau", type=float, default=1.0, help="timescale for stochastic forcing")
    p.add_argument("--ou_var", type=float, default=1.0, help="variance for stochastic forcing")
    p.add_argument("--ou_amp", type=float, default=1, help="amplitude for stochastic forcing")
    p.add_argument("--ou_k_min", type=float, default=2.5, help="minimum |k| excited by OU forcing")
    p.add_argument("--ou_k_max", type=float, default=8.5, help="maximum |k| excited by OU forcing")
    # cadence / options
    p.add_argument("--PlotInterval", type=int, default=10, help="plot interval in time steps")
    p.add_argument("--SaveInterval", type=int, default=10, help="save interval in time steps")
    # RNG / ensemble for initialization
    p.add_argument("--seed", type=int, default=1, help="RNG seed")
    p.add_argument("--MR", type=int, default=None, help="ensemble size to learn modes (default max(1000,MC))")
    p.add_argument("--smallvar", type=int, default=0, help="Make IC variability come from the vortex's radius")
    p.add_argument("--eps_IC_noise", type=float, default=1e-4, help="Make IC variability come from broadband noise")
    # I/O and plotting
    p.add_argument("--outdir", type=str, default="save/DO_Lamb_Oseen", help="output directory")
    p.add_argument("--name", type=str, default=None, help="base name for outputs (auto if None)")
    p.add_argument("--plotIC", type=int, default=0, help="show a plot of the ICs before iterating")
    p.add_argument("--plot", action="store_true", help="show a quick energy plot at the end")
    p.add_argument("--realtime", action="store_true", help="enable live plotting during the run")
    p.add_argument("--threads", type=int, default=1,
                   help="set BLAS/FFT thread count; if unset, keep current environment")
    return p


def _fmt_Tf(Tf: float) -> str:
    r = round(Tf)
    return str(int(r)) if abs(Tf - r) < 1e-8 else str(Tf)


def _set_threads(n_threads: int | None) -> None:
    if n_threads is None:
        return
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "FFTW_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n_threads)


def main():
    args = _build_parser().parse_args()
    _set_threads(args.threads)

    # Heavy imports after thread caps are in env
    import numpy as np
    from .setup import initial_conditions
    from .plotting import DOPlotter
    from .solver import solve_do
    from .params import Params

    # --- Build Params ---
    P = Params(
        Lx=args.Lx, Ly=args.Ly,
        Nx=args.Nx, Ny=args.Ny,
        Tf=args.Tf, dt=args.dt,
        S=args.S, MC=args.MC,
        Re=args.Re, eps_IC_noise=args.eps_IC_noise,
        PlotInterval=args.PlotInterval, SaveInterval=args.SaveInterval,
        forcing=args.forcing, f_amp=args.f_amp,
        forcing_stochastic=args.forcing_stochastic,
        ou_tau=args.ou_tau, ou_var=args.ou_var, ou_amp=args.ou_amp,
        ou_k_min=args.ou_k_min, ou_k_max=args.ou_k_max,
        Seed=args.seed,
    )

    # --- Initialise ---
    MR_eff = args.MR if args.MR is not None else max(1000, args.MC)
    print(f"[init] Building initial ensemble with MR={MR_eff} (MC={P.MC}, S={P.S}) ...", flush=True)
    Uh, Vh, Uih, Vih, YY, Gxh, Gyh = initial_conditions(P, MR=MR_eff, seed=args.seed,
                                                    plotIC=args.plotIC, smallvar=args.smallvar,
                                                    eps_IC_noise=args.eps_IC_noise)

    # --- Solve ---
    print(f"[solve] Advancing to Tf={P.Tf} with dt={P.dt} (Re={P.Re}) ...", flush=True)
    plotter = DOPlotter(P) if args.realtime else None
    res = solve_do(Uh, Vh, Uih, Vih, YY, Gxh, Gyh, P, plotter=plotter)

    # --- Save outputs ---
    run_name = f"N{args.Nx}_T{_fmt_Tf(args.Tf)}_S{args.S}_MC{args.MC}"
    outdir_base = Path(args.outdir)
    data_dir = outdir_base / "runs" / run_name / f"N-{args.forcing_stochastic}" / f"F-{args.forcing}" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if args.forcing_stochastic == "none":
        if args.smallvar == 1:
            outpath = data_dir / f"fields_smallICvar.npz"
        else:
            outpath = data_dir / f"fields_eps_{args.eps_IC_noise}.npz"
    else:
        outpath = data_dir / f"fields_ouamp-{args.ou_amp}.npz"

    np.savez_compressed(
        outpath,
        # Params
        Lx=P.Lx, Ly=P.Ly, Nx=P.Nx, Ny=P.Ny, Tf=P.Tf, dt=P.dt, S=P.S, MC=P.MC,
        Re=P.Re, PlotIntrvl=P.PlotInterval, SaveIntrvl=P.SaveInterval,
        Forcing=P.forcing, Seed=P.Seed,
        # Initial perturbation amplitude
        eps_IC_noise=P.eps_IC_noise,
        # Forcing (spectral)
        Gxh=Gxh, Gyh=Gyh,
        f_amp=args.f_amp,
        # Stochastic forcing
        stochastic=args.forcing_stochastic,
        ou_tau=args.ou_tau, ou_var=args.ou_var, ou_amp=args.ou_amp,
        ou_k_min=args.ou_k_min, ou_k_max=args.ou_k_max,
        half_k_pairs=res.half_k_pairs,
        ou_amplitudes=res.ou_amplitudes,
        # Solver outputs
        T_save=res.T_save, CYY_save=res.CYY_save,
        U_save=res.U_save, V_save=res.V_save,
        Ui_save=res.Ui_save, Vi_save=res.Vi_save,
        W_save=res.W_save, Wi_save=res.Wi_save,
        YY_save=res.YY_save,
        T_plot=res.T_plot, CYY_plot=res.CYY_plot,
    )
    print(f"[save] Wrote {outpath}")

    # --- Optional energy-per-mode plot ---
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.semilogy(res.T_plot, res.CYY_plot[0, :], label="Mean energy")
            for i in range(P.S):
                ax.semilogy(res.T_plot, res.CYY_plot[1 + i, :], label=f"Mode {i+1}")
            ax.set_xlabel("t")
            ax.set_ylabel("Energy")
            ax.set_title("Energy in mean and modes")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            plt.show()
        except Exception as e:
            print(f"error :(")


if __name__ == "__main__":
    main()
