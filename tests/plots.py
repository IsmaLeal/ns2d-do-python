"""
Inputs
------
--inputs can be:
  - a glob: "/.../data/fields_ouamp-*.npz"
  - a single file: "/.../data/fields_ouamp-1000.0.npz"
  - a comma-separated list: "a.npz,b.npz"

Outputs (multi-file case)
-------------------------
Given outdir = ".../plots":

1) Per-sigma subfolders:
   plots/ouamp-10/
   plots/ouamp-100/
   plots/ouamp-1000/
   Each contains per-run plots:
     - energy_neff.pdf
     - variance_spectrum.pdf
     - marginals_YiYj.pdf   (one per requested pair)
     - spatial_uncertainty.pdf

2) In plots/ (root), grids across sigmas:
     - energy_neff_grid.pdf
     - variance_spectrum_grid.pdf
     - spatial_uncertainty_grid.pdf      (rows=sigmas, cols=[mean omega, std omega])
     - marginals_grid_YiYj.pdf           (one per requested pair; rows=sigmas, cols=times)

3) Extra "surprise" summary plot in plots/ (root):
     - uncertainty_spectrum_overlay.pdf
   This is a single, very compact plot: for each sigma, it shows the *scale-dependent*
   uncertainty ratio at a snapshot time:
        R(k) = S_var(k) / (S_mean(k) + tiny)
   where S_mean(k) is the radial spectrum of |omega_mean_hat|^2 and S_var(k) is the radial
   spectrum of Var[omega_hat] computed from DO modes + coeff covariance.
   If half_k_pairs exists, the forcing band is shaded.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm, Normalize
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable


def cov_spectrum_and_neff(CYY, t, stride=1):
    """
    Returns:
      t_idx: (Tn,)
      lam  : (Tn,S) eigenvalues sorted desc
      neff : (Tn,)
    """
    _, T = CYY.shape
    idx = np.arange(0, T, max(1, stride))
    # sorted eigenvalues
    lam = CYY[1:, idx].T    # (idx,S)
    lam = np.sort(lam, axis=1)[:, ::-1] # sort descending per timestep
    # effective number of
    s1 = lam.sum(axis=1)
    s2 = np.sum(lam*lam, axis=1)
    neff = np.where(s2 > 1e-30, (s1*s1)/s2, 0.0)

    return t[idx], lam, neff


def energy_partition_CYY(CYY):
    """
    CYY: (1+S, T) where CYY[0]=mean energy, others are mode energies.
    Returns E_tot, E_mean, E_stoch (all (T,))
    """
    E_mean = CYY[0, :].copy()
    E_stoch = np.sum(CYY[1:, :], axis=0)
    E_tot = E_mean + E_stoch
    return E_tot, E_mean, E_stoch


def choose_snapshot_index(T, window_frac, mode):
    w0 = int(np.floor(window_frac * T))
    if mode == "window_start":
        return w0
    if mode == "window_mid":
        return w0 + (T - 1 - w0) // 2
    if mode == "final":
        return T - 1
    return w0


def fft_wavenumbers(Nx, Ny, Lx, Ly):
    dx = Lx / Nx
    dy = Ly / Ny
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX*KX + KY*KY)
    return KX, KY, K


def forcing_band_from_half_k_pairs(data, Lx, Ly):
    """
    Returns (kf_min, kf_max) from half_k_pairs if present, else (None,None).
    """
    if "half_k_pairs" not in data:
        return None, None
    hk = np.asarray(data["half_k_pairs"], dtype=int).reshape(-1, 2)
    if hk.size == 0:
        return None, None

    ix = hk[:, 0].astype(float)
    iy = hk[:, 1].astype(float)
    kx = 2*np.pi * ix / float(Lx)
    ky = 2*np.pi * iy / float(Ly)
    kmag = np.sqrt(kx*kx + ky*ky)
    return float(kmag.min()), float(kmag.max())


def get_radial_binner(run, nbins):
    """
    Cache bin ids for radial shell sums (depends only on K and nbins).
    Returns (k_cent, bin_id, nbins)
    """
    cache = run["_radial_cache"]
    if nbins in cache:
        return cache[nbins]

    K = run["K"]
    k_flat = K.ravel()
    kmax = float(k_flat.max())
    edges = np.linspace(0.0, kmax, nbins + 1)
    bin_id = np.digitize(k_flat, edges) - 1
    bin_id = np.clip(bin_id, 0, nbins - 1)

    k_cent = 0.5 * (edges[:-1] + edges[1:])
    cache[nbins] = (k_cent, bin_id, nbins)
    return cache[nbins]


def radial_shell_sum_fast(A2d, bin_id, nbins):
    """
    A2d: (Ny,Nx) field in k-space to bin.
    """
    return np.bincount(bin_id, weights=A2d.ravel(), minlength=nbins).astype(float)


# ------------------------- styling ------------------------- #
def setup_style():
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update({
        "font.size": 13,
        "legend.fontsize": 18,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "font.family": "serif"
    })


def sigma_folder_name(sigma: float):
    return f"ouamp-{sigma:g}"


# ------------------------- plot: energy split + Neff ------------------------- #
def plot_energy_neff(ax, run):
    ax2 = ax.twinx()    # create a second, right y-axis for the same x-axis

    lEt, = ax.plot(run["t"], run["E_tot"],  lw=2, label=r"$E_{\mathrm{tot}}$", color="k")
    lEm, = ax.plot(run["t"], run["E_mean"], lw=1.4, ls="--", label=r"$E_{\mathrm{mean}}$", color="lightseagreen")
    lEs, = ax.plot(run["t"], run["E_stoch"], lw=1.6, ls="--",  label=r"$E_{\mathrm{stoch}}$", color="b")

    lN, = ax2.plot(run["t_neff"], run["Neff"], lw=2, ls=":", color="sienna",
                   label=r"$N_{\mathrm{eff}}$") #, marker="o", ms=1.5)

    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax2.set_ylabel(r"$N_{\mathrm{eff}}$", color="sienna")
    ax2.tick_params(axis="y", colors="sienna")
    ax2.spines["right"].set_color("sienna")

    ax.grid(True, alpha=0.25)
    ax2.grid(False)

    lines = [lEt, lEm, lEs, lN]
    labels = [L.get_label() for L in lines]
    ax.legend(lines, labels, frameon=False, loc="upper right")


def save_energy_neff_single(run, outpath):
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 4.0))
    plot_energy_neff(ax, run)
    fig.suptitle(rf"Energy split & $N_{{\mathrm{{eff}}}}$ for $\sigma={run['sigma']:g}$", y=0.95)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_energy_neff_grid(runs, outpath):
    runs = sorted(runs, key=lambda r: r["sigma"])
    n = len(runs)
    if n == 1:
        save_energy_neff_single(runs[0], outpath)
        return
    if n == 2:
        nrows, ncols = 1, 2
    elif n <= 4:
        nrows, ncols = 2, 2
    else:
        ncols = 3
        nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4*ncols, 3.8*nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")

    first_ax = None
    for i, run in enumerate(runs):
        ax = axes[i]
        if first_ax is None:
            first_ax = ax
        plot_energy_neff(ax, run)
        ax.set_title(rf"$\sigma={run['sigma']:g}$", fontsize=16)
        ax.legend().set_visible(False)

    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(labels),
               frameon=False)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ------------------------- plot: variance spectrum heatmap ------------------------- #
def save_variance_spectrum_single(run, outpath, heatmap_log=False):
    """
    Heatmap of lambda_i(t)/sum lambda.
    """
    t = run["t_neff"]
    lam = run["lam_neff"]  # (Tn,S)

    Z = lam / np.maximum(lam.sum(axis=1, keepdims=True), 1e-30) # (T,S) array with fractional variance for each mode and timestep
    if heatmap_log:
        Z = np.log10(np.maximum(Z, 1e-30))

    fig, ax = plt.subplots(1, 1, figsize=(8.6, 3.8))
    im = ax.imshow(
        Z.T,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 1, Z.shape[1]],
        interpolation="nearest",
        #vmin=np.min(Z),
        #vmax=np.max(Z),
        norm=PowerNorm(gamma=0.5, vmin=0.0, vmax=np.max(Z))
    )
    ax.grid(False)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mode index")
    ttl = rf"Explained variance per mode and time for $\sigma={run['sigma']:g}$"
    if heatmap_log:
        ttl += " (log10)"
    ax.set_title(ttl)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("value")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_variance_spectrum_grid(runs, outpath, heatmap_log=False, gamma=0.5):
    runs = sorted(runs, key=lambda r: r["sigma"])
    n = len(runs)

    if n == 1:
        save_variance_spectrum_single(runs[0], outpath, heatmap_log=heatmap_log)
        return

    # ---- build Z(t,i) for each sigma (same as single) ----
    Z_list, t_list, zmax_list, zmin_list = [], [], [], []
    for r in runs:
        t = r["t_neff"]
        lam = r["lam_neff"]  # (Tn,S)
        Z = lam / np.maximum(lam.sum(axis=1, keepdims=True), 1e-30)
        if heatmap_log:
            Z = np.log10(np.maximum(Z, 1e-30))
        Z_list.append(Z)
        t_list.append(t)
        zmax_list.append(float(np.max(Z)))
        zmin_list.append(float(np.min(Z)))

    # ---- ABS column: shared LINEAR scale across all sigmas ----
    Z_all = np.concatenate([Z.ravel() for Z in Z_list])
    vmin_abs = float(np.min(Z_all))
    vmax_abs = float(np.max(Z_all))
    if not heatmap_log:
        vmin_abs = 0.0
    norm_abs = Normalize(vmin=vmin_abs, vmax=vmax_abs)

    # ---- figure layout: 2 plot columns with SHARED ABS colorbar BETWEEN them ----
    fig = plt.figure(figsize=(12.0, 2.8 * n))
    gs = fig.add_gridspec(
        nrows=n, ncols=3,
        width_ratios=[1.0, 0.05, 1.0],   # <-- ONLY change: colorbar col in the middle
        wspace=0.28, hspace=0.38
    )
    cax_abs = fig.add_subplot(gs[:, 1])  # <-- ONLY change: now between ABS and REL

    im_abs_ref = None

    for i, r in enumerate(runs):
        ax_abs = fig.add_subplot(gs[i, 0])
        ax_rel = fig.add_subplot(gs[i, 2])  # <-- ONLY change: REL moved to col 2

        t = t_list[i]
        Z = Z_list[i]
        S = Z.shape[1]

        # --- ABS (shared linear) ---
        im_abs = ax_abs.imshow(
            Z.T, aspect="auto", origin="lower",
            extent=[t[0], t[-1], 1, S],
            interpolation="nearest",
            norm=norm_abs
        )
        if im_abs_ref is None:
            im_abs_ref = im_abs

        ax_abs.grid(False)
        ax_abs.set_ylabel("Mode index")
        ax_abs.set_title(rf"$\sigma={r['sigma']:g}$ (shared colormap)")
        ax_abs.set_xlabel("Time" if i == n - 1 else "")

        # --- REL (per-sigma scale + per-sigma colorbar) ---
        if heatmap_log:
            norm_rel = Normalize(vmin=zmin_list[i], vmax=zmax_list[i])
        else:
            norm_rel = PowerNorm(gamma=gamma, vmin=0.0, vmax=zmax_list[i])

        im_rel = ax_rel.imshow(
            Z.T, aspect="auto", origin="lower",
            extent=[t[0], t[-1], 1, S],
            interpolation="nearest",
            norm=norm_rel
        )

        ax_rel.grid(False)
        ax_rel.set_title(rf"$\sigma={r['sigma']:g}$")
        ax_rel.set_xlabel("Time" if i == n - 1 else "")

        # per-sigma colorbar for REL
        divider = make_axes_locatable(ax_rel)
        cax_rel = divider.append_axes("right", size="3.5%", pad=0.08)
        cb_rel = fig.colorbar(im_rel, cax=cax_rel)
        cb_rel.set_label("value")

    # shared colorbar ONLY for ABS column
    cb_abs = fig.colorbar(im_abs_ref, cax=cax_abs)

    fig.suptitle("Explained variance per mode and time", y=0.995)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ------------------------- plot: energy per wavenumber plot ------------------------- #
def compute_Ek_stats(run, window_frac=0.5, stride=5, nbins=40, qlo=0.16, qhi=0.84):
    """
    Time-windowed stats for E(k).
    Returns:
      k_cent
      Et_mean, Em_mean, Es_mean
      Et_lo, Et_hi   (quantile band across time, default 16-84%)
    """
    t = run["t"]
    T = t.size
    w0 = int(np.floor(window_frac * T))
    idxs = np.arange(w0, T, max(1, stride))

    Nx, Ny = run["Nx"], run["Ny"]
    fac = 1.0 / (Nx * Ny)**2

    U_save, V_save = run["U_save"], run["V_save"]
    Ui_save, Vi_save = run["Ui_save"], run["Vi_save"]
    lamY = run["CYY"][1:, :]  # (S,T) diag variances

    k_cent, bin_id, nbins = get_radial_binner(run, nbins)

    Et_list = []
    Em_list = []
    Es_list = []

    for k in idxs:
        # mean spectrum
        uhat = np.fft.fft2(U_save[..., k])
        vhat = np.fft.fft2(V_save[..., k])
        Ehat_mean = 0.5 * (np.abs(uhat)**2 + np.abs(vhat)**2) * fac

        # stochastic (diag covariance)
        lam_k = lamY[:, k]  # (S,)
        Ui_k = Ui_save[..., k]  # (Ny,Nx,S) view
        Vi_k = Vi_save[..., k]

        Ui_hat = np.fft.fft2(Ui_k, axes=(0, 1))
        Vi_hat = np.fft.fft2(Vi_k, axes=(0, 1))

        var_u_hat = np.sum((np.abs(Ui_hat)**2) * lam_k[None, None, :], axis=2)
        var_v_hat = np.sum((np.abs(Vi_hat)**2) * lam_k[None, None, :], axis=2)
        Ehat_stoch = 0.5 * (var_u_hat + var_v_hat) * fac

        Em = radial_shell_sum_fast(Ehat_mean,  bin_id, nbins)
        Es = radial_shell_sum_fast(Ehat_stoch, bin_id, nbins)
        Et = Em + Es

        Em_list.append(Em)
        Es_list.append(Es)
        Et_list.append(Et)

    Em_arr = np.asarray(Em_list)  # (Nt,nbins)
    Es_arr = np.asarray(Es_list)
    Et_arr = np.asarray(Et_list)

    Em_mean = Em_arr.mean(axis=0)
    Es_mean = Es_arr.mean(axis=0)
    Et_mean = Et_arr.mean(axis=0)

    Et_lo = np.quantile(Et_arr, qlo, axis=0)
    Et_hi = np.quantile(Et_arr, qhi, axis=0)

    return k_cent, Et_mean, Em_mean, Es_mean, Et_lo, Et_hi


def save_Ek_single(run, outpath, window_frac=0.5, stride=5, nbins=40,
                   xscale="linear", yscale="log", shade_forcing=True, shade_uncert=True):
    k, Et, Em, Es, Et_lo, Et_hi = compute_Ek_stats(
        run, window_frac=window_frac, stride=stride, nbins=nbins
    )

    # log plots need positive
    mask = (k > 0) & (Et > 0) & (Em > 0) & (Es > 0)
    k, Et, Em, Es, Et_lo, Et_hi = k[mask], Et[mask], Em[mask], Es[mask], Et_lo[mask], Et_hi[mask]

    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.0))

    if yscale == "log":
        ax.set_yscale("log")
    if xscale == "log":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.set_xscale("linear")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

    ax.plot(k, Et, lw=2.2, color="k", label=r"$E_{\mathrm{tot}}(k)$")
    ax.plot(k, Em, lw=1.6, ls="--", color="lightseagreen", label=r"$E_{\mathrm{mean}}(k)$")
    ax.plot(k, Es, lw=1.6, ls="--", color="b", label=r"$E_{\mathrm{stoch}}(k)$")

    # --- uncertainty band (use percentiles, not mean±std) ---
    if shade_uncert:
        # ignore bins that are effectively numerical noise
        floor = 1e-30 * np.max(Et)
        mband = (Et_lo > floor) & (Et_hi > floor)
        ax.fill_between(k[mband], Et_lo[mband], Et_hi[mband],
                        alpha=0.8, color="0.6", label=r"$[16,84]\%$ across time")

    if shade_forcing:
        Lx, Ly = run["Lx"], run["Ly"]
        kf_min, kf_max = forcing_band_from_half_k_pairs(run["data"], Lx, Ly)
        if (kf_min is not None) and (kf_max is not None):
            ax.axvspan(kf_min, kf_max, alpha=0.12, color="k", label="forcing band")

    # --- IMPORTANT: set sane y-lims so band can't nuke the plot ---
    pos = Et[Et > 0]
    ymin = max(np.min(pos), 1e-30 * np.max(pos))
    ymax = max(np.max(pos), np.max(Et_hi))  # allow upper band
    ax.set_ylim(ymin, 1.3 * ymax)

    ax.grid(True, alpha=0.2)
    ax.set_xlabel(r"$|k|$")
    ax.set_ylabel(r"$E(k)$ (shell-sum, time-window)")
    ax.set_title(rf"$E(k)$ for $\sigma={run['sigma']:g}$ (avg from {window_frac:.2f}T)")

    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_Ek_grid(
    runs, outpath,
    window_frac=0.5, stride=5, nbins=40,
    xscale="linear", yscale="log",      # <-- choose "linear"/"log"
    shade_forcing=True, shade_uncert=True,
    qlo=0.16, qhi=0.84,
    share_ylim=False                    # True = comparable amplitudes across sigmas
):
    runs = sorted(runs, key=lambda r: r["sigma"])
    n = len(runs)

    # layout
    if n == 1:
        save_Ek_single(runs[0], outpath, xscale=="linear", yscale=="log", window_frac=window_frac, stride=stride, nbins=nbins,
                       shade_forcing=shade_forcing, shade_uncert=shade_uncert)
        return
    if n == 2:
        nrows, ncols = 1, 2
    elif n <= 4:
        nrows, ncols = 2, 2
    else:
        ncols = 3
        nrows = int(np.ceil(n / ncols))

    # forcing band (same for all runs; use first run that has it)
    forcing_band = None
    if shade_forcing:
        for r in runs:
            kf_min, kf_max = forcing_band_from_half_k_pairs(r["data"], r["Lx"], r["Ly"])
            if (kf_min is not None) and (kf_max is not None):
                forcing_band = (kf_min, kf_max)
                break

    # precompute all spectra (mean + stoch + tot + percentile band)
    specs = []
    global_ymin = np.inf
    global_ymax = 0.0

    for r in runs:
        k, Et, Em, Es, Et_lo, Et_hi = compute_Ek_stats(
            r, window_frac=window_frac, stride=stride, nbins=nbins, qlo=qlo, qhi=qhi
        )

        # masks for log scales
        m = np.isfinite(k) & np.isfinite(Et) & (k >= 0)
        if xscale == "log":
            m &= (k > 0)
        if yscale == "log":
            m &= (Et > 0) & (Em > 0) & (Es > 0) & (Et_hi > 0)

        k, Et, Em, Es, Et_lo, Et_hi = k[m], Et[m], Em[m], Es[m], Et_lo[m], Et_hi[m]

        # band mask: ignore numerical-noise bins so it can't destroy y-lims
        floor = 1e-30 * np.max(Et) if Et.size else 0.0
        band_mask = (Et_lo > floor) & (Et_hi > floor)

        # y-lims (same logic as single)
        if Et.size:
            pos = Et[Et > 0]
            ymin = max(np.min(pos), 1e-12 * np.max(pos)) if (yscale == "log") else float(np.min(Et))
            ymax = max(np.max(pos), np.max(Et_hi))       if (yscale == "log") else float(np.max(Et_hi))
            global_ymin = min(global_ymin, ymin)
            global_ymax = max(global_ymax, ymax)

        specs.append((k, Et, Em, Es, Et_lo, Et_hi, band_mask))

    # figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6*ncols, 3.8*nrows))
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")

    first_ax = None

    for i, r in enumerate(runs):
        ax = axes[i]
        if first_ax is None:
            first_ax = ax

        k, Et, Em, Es, Et_lo, Et_hi, band_mask = specs[i]

        # scales
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        # plot (only label first subplot for global legend)
        lab_tot   = r"$E_{\mathrm{tot}}(k)$"   if i == 0 else "_nolegend_"
        lab_mean  = r"$E_{\mathrm{mean}}(k)$"  if i == 0 else "_nolegend_"
        lab_stoch = r"$E_{\mathrm{stoch}}(k)$" if i == 0 else "_nolegend_"
        lab_band  = rf"$[{int(100*qlo)},{int(100*qhi)}]\%$ across time" if i == 0 else "_nolegend_"

        ax.plot(k, Et, lw=2.2, color="k", label=lab_tot)
        ax.plot(k, Em, lw=1.6, ls="--", color="lightseagreen", label=lab_mean)
        ax.plot(k, Es, lw=1.6, ls="--", color="b", label=lab_stoch)

        if shade_uncert and np.any(band_mask):
            ax.fill_between(k[band_mask], Et_lo[band_mask], Et_hi[band_mask],
                            alpha=0.8, color="0.6", label=lab_band)

        if forcing_band is not None:
            kf_min, kf_max = forcing_band
            ax.axvspan(kf_min, kf_max, alpha=0.12, color="k",
                       label=("forcing band" if i == 0 else "_nolegend_"))

        ax.set_title(rf"$\sigma={r['sigma']:g}$")
        ax.set_xlabel(r"$|k|$")
        ax.set_ylabel(r"$E(k)$")
        ax.grid(True, alpha=0.2)

        if xscale == "log":
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=6))
            ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        else:
            ax.set_xscale("linear")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))

        # y-lims
        if share_ylim and np.isfinite(global_ymin) and global_ymax > 0:
            ax.set_ylim(global_ymin, 1.3*global_ymax)
        else:
            # per-panel sane y-lims
            if Et.size:
                if yscale == "log":
                    pos = Et[Et > 0]
                    ymin = max(np.min(pos), 1e-12*np.max(pos))
                    ymax = max(np.max(pos), np.max(Et_hi))
                    ax.set_ylim(ymin, 1.3*ymax)

    # one legend for the whole grid
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, 0.97))

    fig.suptitle(rf"$E(k)$ across sigmas (avg from {window_frac:.2f}T)", y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ------------------------- plot: 2D marginals grids ------------------------- #
def parse_pairs(pairs_str, S):
    """
    pairs_str like "1,2;1,3" (1-based)
    """
    pairs = []
    for chunk in [c.strip() for c in pairs_str.split(";") if c.strip()]:
        a, b = [int(x.strip()) for x in chunk.split(",")]
        i, j = a - 1, b - 1
        if 0 <= i < S and 0 <= j < S and i != j:
            pairs.append((i, j))
    if not pairs:
        pairs = [(0, 1)]
    return pairs


def choose_time_indices(T, fracs):
    idx = []
    for f in fracs:
        k = int(np.clip(np.floor(f * (T - 1)), 0, T - 1))
        idx.append(k)
    out = []
    for k in idx:
        if k not in out:
            out.append(k)
    return out


def robust_limits(all_xy, q=0.995):
    """
    all_xy: list of arrays (MC,2) or (N,2)
    Returns symmetric limits (xmin,xmax,ymin,ymax) using quantiles.
    """
    XY = np.vstack(all_xy)
    x = XY[:, 0]
    y = XY[:, 1]
    # symmetric around 0 using quantile of abs (gives stable comparisons across sigmas)
    ax = np.quantile(np.abs(x), q) + 1e-12
    ay = np.quantile(np.abs(y), q) + 1e-12
    return (-ax, ax, -ay, ay)


def save_marginals_single(run, outpath, pair, time_idxs):
    """
    One sigma, one pair, multiple times (columns).
    """
    i, j = pair
    YY = run["YY"]  # (MC,S,T)
    t = run["t"]
    MC, S, T = YY.shape

    # consistent limits across the chosen times (within this sigma)
    all_xy = []
    for k in time_idxs:
        all_xy.append(np.c_[YY[:, i, k], YY[:, j, k]])
    x0, x1, y0, y1 = robust_limits(all_xy, q=0.995)

    fig, axes = plt.subplots(1, len(time_idxs), figsize=(3.4*len(time_idxs), 3.2), squeeze=False)
    for c, k in enumerate(time_idxs):
        ax = axes[0, c]
        ax.scatter(YY[:, i, k], YY[:, j, k], s=10, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.grid(True, alpha=0.2)
        ax.set_title(rf"$t={t[k]:.3g}$")
        ax.set_xlabel(rf"$Y_{{{i+1}}}$")
        if c == 0:
            ax.set_ylabel(rf"$Y_{{{j+1}}}$")

    fig.suptitle(rf"$\sigma={run['sigma']:g}$   2D marginals  $(Y_{i+1},Y_{j+1})$", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_marginals_grid(runs, outpath, pair, time_idxs):
    """
    Rows = sigmas, Cols = times (for one pair).
    Shared axis limits across all sigmas and selected times.
    """
    runs = sorted(runs, key=lambda r: r["sigma"])
    i, j = pair

    all_xy = []
    for r in runs:
        YY = r["YY"]
        for k in time_idxs:
            all_xy.append(np.c_[YY[:, i, k], YY[:, j, k]])
    x0, x1, y0, y1 = robust_limits(all_xy, q=0.995)

    nrows = len(runs)
    ncols = len(time_idxs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 3.0*nrows), squeeze=False)

    for rr, r in enumerate(runs):
        YY = r["YY"]
        t = r["t"]
        for cc, k in enumerate(time_idxs):
            ax = axes[rr, cc]
            ax.scatter(YY[:, i, k], YY[:, j, k], s=8, alpha=0.22)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.grid(True, alpha=0.18)

            if rr == 0:
                ax.set_title(rf"$t={t[k]:.3g}$")
            if cc == 0:
                ax.set_ylabel(rf"$\sigma={r['sigma']:g}$" + "\n" + rf"$Y_{{{j+1}}}$")
            if rr == nrows - 1:
                ax.set_xlabel(rf"$Y_{{{i+1}}}$")

    fig.suptitle(rf"2D marginals grid for $(Y_{i+1},Y_{j+1})$ (all sigmas)", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ------------------------- plot: spatial uncertainty grids ------------------------- #
def compute_spatial_uncertainty_omega(data, snapshot_k, YY=None):
    """
    Returns:
      omega_mean (Ny,Nx)
      omega_std  (Ny,Nx)
    Var[omega(x)] = Wi(x,:) C Wi(x,:)^T
    """
    w_mean = np.array(data["W_save"][..., snapshot_k], dtype=float)     # (Ny,Nx)
    Wi = np.array(data["Wi_save"][..., snapshot_k], dtype=float)        # (Ny,Nx,S)

    YY = np.array(YY[:, :, snapshot_k], dtype=float)       # (MC,S)
    MC = YY.shape[0]
    Ck = (YY.T @ YY) / MC

    tmp = np.tensordot(Wi, Ck, axes=(2, 0))    # (Ny,Nx,S)
    var = np.sum(tmp * Wi, axis=2)             # (Ny,Nx)
    var[var < 0] = 0.0
    std = np.sqrt(var)
    return w_mean, std


def save_spatial_uncertainty_single(run, outpath, snapshot_k, YY=None, q_mean=0.99, q_std=0.99):
    data = run["data"]
    w_mean, w_std = compute_spatial_uncertainty_omega(
        data, snapshot_k, YY=YY
    )

    # robust per-sigma limits
    mlim = np.quantile(np.abs(w_mean.ravel()), q_mean) + 1e-12
    vmin_mean, vmax_mean = -mlim, mlim

    vmax_std = np.quantile(w_std.ravel(), q_std) + 1e-12
    vmin_std = 0.0

    Lx = float(data["Lx"])
    Ly = float(data["Ly"])
    extent = [0, Lx, 0, Ly]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    im0 = axes[0].imshow(w_mean, origin="lower", extent=extent,
                         vmin=vmin_mean, vmax=vmax_mean)
    axes[0].set_title(rf"Mean vorticity $\bar\omega$ at $t={run['t'][snapshot_k]:.3g}$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)

    im1 = axes[1].imshow(w_std, origin="lower", extent=extent,
                         vmin=vmin_std, vmax=vmax_std)
    axes[1].set_title(rf"Std vorticity $\sqrt{{\mathrm{{Var}}(\omega)}}$ at $t={run['t'][snapshot_k]:.3g}$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.suptitle(rf"Spatial mean and uncertainty for $\sigma={run['sigma']:g}$", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def save_spatial_uncertainty_grid(runs, outpath, snapshot_k,
                                  q_mean=0.99, q_std_vmin=0.01, q_std_vmax=0.99):
    """
    Rows = sigmas; Cols = [mean omega, std omega]
    snapshot_k is an *index* (same convention as save_spatial_uncertainty_single()).

    Mean: shared symmetric linear limits across sigmas.
    Std : shared LogNorm limits across sigmas (colorbar shows std values).
    """
    runs = sorted(runs, key=lambda r: r["sigma"])
    nrows = len(runs)

    means, stds, t_used = [], [], []

    # --- use snapshot_k as INDEX, consistently ---
    for r in runs:
        t = r["t"]
        T = len(t)
        k = int(snapshot_k)
        if k < 0:
            k = T + k
        if not (0 <= k < T):
            raise IndexError(f"snapshot_k={snapshot_k} out of bounds for T={T}")

        t_used.append(float(t[k]))
        w_mean, w_std = compute_spatial_uncertainty_omega(r["data"], k, YY=r["YY"])
        means.append(w_mean)
        stds.append(w_std)

    # ---- shared limits (mean) ----
    all_mean = np.hstack([m.ravel() for m in means])
    mlim = np.quantile(np.abs(all_mean), q_mean) + 1e-12
    vmin_mean, vmax_mean = -mlim, mlim

    # ---- shared limits (std, positive) ----
    all_std = np.hstack([s.ravel() for s in stds])
    vmin_std = np.quantile(all_std, q_std_vmin)
    vmax_std = np.quantile(all_std, q_std_vmax)
    vmin_std = max(float(vmin_std), 1e-12)
    vmax_std = max(float(vmax_std), vmin_std * 10.0)
    norm_std = LogNorm(vmin=vmin_std, vmax=vmax_std)

    # ---- layout: [mean | cbar_mean | std | cbar_std] ----
    fig = plt.figure(figsize=(10.6, 2.9 * nrows))
    gs = fig.add_gridspec(
        nrows=nrows, ncols=4,
        width_ratios=[1.0, 0.045, 1.0, 0.045],
        wspace=0.25, hspace=0.25
    )

    ax_mean = [fig.add_subplot(gs[i, 0]) for i in range(nrows)]
    ax_std  = [fig.add_subplot(gs[i, 2]) for i in range(nrows)]
    cax_mean = fig.add_subplot(gs[:, 1])
    cax_std  = fig.add_subplot(gs[:, 3])

    im_mean0 = None
    im_std0 = None

    for i, r in enumerate(runs):
        data = r["data"]
        extent = [0, float(data["Lx"]), 0, float(data["Ly"])]

        im0 = ax_mean[i].imshow(means[i], origin="lower", extent=extent,
                                vmin=vmin_mean, vmax=vmax_mean)
        im1 = ax_std[i].imshow(stds[i], origin="lower", extent=extent,
                               norm=norm_std)

        if im_mean0 is None: im_mean0 = im0
        if im_std0 is None: im_std0 = im1

        ax_mean[i].set_ylabel(rf"$\sigma={r['sigma']:g}$" + "\n" + "y")
        ax_mean[i].set_xlabel("x")
        ax_std[i].set_xlabel("x")
        ax_std[i].set_ylabel("y")

        if i == 0:
            ax_mean[i].set_title(rf"mean $\bar\omega$ (k={int(snapshot_k)}, t≈{t_used[i]:.3g})")
            ax_std[i].set_title(
                rf"std $\sqrt{{\mathrm{{Var}}(\omega)}}$ (LogNorm; k={int(snapshot_k)}, t≈{t_used[i]:.3g})"
            )

    cb0 = fig.colorbar(im_mean0, cax=cax_mean)
    cb0.set_label(r"mean vorticity $\bar\omega$")

    cb1 = fig.colorbar(im_std0, cax=cax_std)
    cb1.set_label(r"std vorticity $\sqrt{\mathrm{Var}(\omega)}$  [log color scale]")
    cb1.locator = mticker.LogLocator(base=10)
    cb1.formatter = mticker.LogFormatterSciNotation(base=10)
    cb1.update_ticks()

    fig.suptitle("Spatial uncertainty grid", y=0.995)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ------------------------- main ------------------------- #
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--inputs", required=True,
                    help="Glob or comma-separated list of .npz files.")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: two levels above first input, then /plots).")

    ap.add_argument("--neff_stride", type=int, default=3,
                    help="Stride for Neff time series (from YY covariance eigs).")
    ap.add_argument("--spec_stride", type=int, default=3,
                    help="Stride for variance-spectrum heatmap (lambda_i(t)).")

    ap.add_argument("--window_frac", type=float, default=0.5,
                    help="Window start fraction used for snapshot selection.")
    ap.add_argument("--snapshot_mode", type=str, default="window_mid",
                    choices=["window_start", "window_mid", "final"],
                    help="Which time to use for spatial uncertainty and uncertainty-spectrum overlay.")

    ap.add_argument("--marg_fracs", type=str, default="0.15,0.55,0.9",
                    help="Fractions of time for marginals, comma-separated.")
    ap.add_argument("--marg_pairs", type=str, default="1,2;1,3;2,3",
                    help='Pairs like "1,2;1,3" (1-based).')

    ap.add_argument("--heatmap_log", action="store_true",
                    help="Plot log10 of variance-spectrum heatmap fractions.")

    ap.add_argument("--spec_logy", dest="spec_logy", action="store_true",
                    help="Use log y-scale for uncertainty-spectrum overlay.")
    ap.add_argument("--no-spec_logy", dest="spec_logy", action="store_false")
    ap.set_defaults(spec_logy=False)

    ap.add_argument("--spec_nbins", type=int, default=30,
                    help="Number of radial bins for uncertainty-spectrum overlay.")

    args = ap.parse_args()

    # get ordered input .npz files
    parts = [p.strip() for p in args.inputs.split(",")]
    files = []
    for p in parts:     # find Unix pattern matches
        if any(ch in p for ch in ["*", "?", "["]):
            files += list(Path().glob(p))
        else:
            files.append(Path(p))
    files = [f for f in files if f.suffix == ".npz"]
    if not files:
        raise FileNotFoundError("No .npz inputs found.")
    files = sorted(files)
    # output directory
    outdir = Path(args.outdir) if args.outdir else (files[0].parents[1] / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    # plotting style and fontsizes
    setup_style()

    # load and precompute run objects
    runs = []
    for f in files:
        data = np.load(f, allow_pickle=True)

        # stochastic forcing amplitude and time values
        sigma = float(data["ou_amp"])
        if int(sigma) == 1:
            continue
        t = np.array(data["T_save"], dtype=float)       # (T,)

        YY = np.array(data["YY_save"], dtype=float)     # (MC,S,T)
        CYY = np.array(data["CYY_save"], dtype=float)   # (S+1,T)
        U_save = np.asarray(data["U_save"])  # (Ny,Nx,T)
        V_save = np.asarray(data["V_save"])
        Ui_save = np.asarray(data["Ui_save"])  # (Ny,Nx,S,T)
        Vi_save = np.asarray(data["Vi_save"])

        Lx = float(data["Lx"])
        Ly = float(data["Ly"])
        Ny, Nx, T = U_save.shape

        KX, KY, K = fft_wavenumbers(Nx, Ny, Lx, Ly)

        # stored energies per timestep
        E_tot, E_mean, E_stoch = energy_partition_CYY(CYY)

        # covariance spectrum + N_eff
        t_neff, lam_neff, Neff = cov_spectrum_and_neff(CYY, t, stride=args.neff_stride)

        run = dict(
            file=str(f),
            data=data,
            sigma=sigma,
            t=t,
            U_save=U_save,
            V_save=V_save,
            Ui_save=Ui_save,
            Vi_save=Vi_save,
            Lx=Lx, Ly=Ly,
            Nx=Nx, Ny=Ny,
            K=K,
            E_tot=E_tot,
            E_mean=E_mean,
            E_stoch=E_stoch,
            t_neff=t_neff,
            lam_neff=lam_neff,
            Neff=Neff,
            YY=YY,
            CYY=CYY,
            _radial_cache={}
        )
        runs.append(run)
    print("All files loaded!")
    # snapshot index shared across sigmas (same time index in saved arrays)
    T0 = runs[0]["t"].size  # (assumes same T across runs)
    snapshot_k = choose_snapshot_index(T0, args.window_frac, args.snapshot_mode)

    # per-sigma folders
    for run in runs:
        print(rf"Producing + saving plots for $\sigma=${run["sigma"]}")
        # create folder if not existing
        sigdir = outdir / sigma_folder_name(run["sigma"])
        sigdir.mkdir(parents=True, exist_ok=True)

        save_energy_neff_single(run, sigdir / "energy_neff.pdf")
        save_variance_spectrum_single(run, sigdir / "variance_spectrum.pdf", heatmap_log=args.heatmap_log)
        # save_Ek_single(run, sigdir / "Ek.pdf",
        #                window_frac=args.window_frac, stride=5, nbins=40,
        #                shade_forcing=True)

        # marginals: one PDF per pair, with multiple times
        MC, S, T = run["YY"].shape
        pairs = parse_pairs(args.marg_pairs, S)
        fracs = [float(x.strip()) for x in args.marg_fracs.split(",") if x.strip()]
        time_idxs = choose_time_indices(T, fracs)
        for (i, j) in pairs:
            save_marginals_single(run, sigdir / f"marginals_Y{i+1}_Y{j+1}.pdf", (i, j), time_idxs)

        save_spatial_uncertainty_single(run, sigdir / "spatial_uncertainty.pdf",
                                        snapshot_k=snapshot_k, YY=run["YY"])

    # ---------------- root grids across sigmas ---------------- #
    save_energy_neff_grid(runs, outdir / "energy_neff_grid_LARGE.pdf")
    save_variance_spectrum_grid(runs, outdir / "variance_spectrum_grid_LARGE.pdf", heatmap_log=args.heatmap_log)
    save_Ek_grid(runs, outdir / "Ek_grid_LARGE.pdf",
                 window_frac=args.window_frac, stride=5, nbins=40, shade_forcing=True)
    save_spatial_uncertainty_grid(runs, outdir / "spatial_uncertainty_grid_LARGE.pdf",
                                  snapshot_k=snapshot_k)

    # marginals grids: one per pair
    MC, S, T = runs[0]["YY"].shape
    pairs = parse_pairs(args.marg_pairs, S)
    fracs = [float(x.strip()) for x in args.marg_fracs.split(",") if x.strip()]
    time_idxs = choose_time_indices(T, fracs)
    for (i, j) in pairs:
        save_marginals_grid(runs, outdir / f"marginals_grid_Y{i+1}_Y{j+1}_LARGE.pdf", (i, j), time_idxs)

    print("Saved to:", outdir)
    print("Per-sigma folders: ouamp-*/")


if __name__ == "__main__":
    main()
