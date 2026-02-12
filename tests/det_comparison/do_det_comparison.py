import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

# ==== Paths ====
DET_PATH = "../../save/DO_Lamb_Oseen/deterministic_comparison/N64_T10_S0_MC200/N-none/F-kolmogorov/data/N64_T10.0_kolmogorov_dtover10.npz"
DO_PATH  = "../../save/DO_Lamb_Oseen/deterministic_comparison/N64_T10_S0_MC200/N-none/F-kolmogorov/data/fields_det_dt.npz"

# ==== Load ====
z_det = np.load(DET_PATH, allow_pickle=True)
z_do  = np.load(DO_PATH,  allow_pickle=True)

what = 'snapshots'    # 'tests'; 'res' residual velocity comparison; 'snapshots'

if what == 'tests':
    # Deterministic (for grid only)
    x, y  = z_det["x"], z_det["y"]          # (Nx,), (Ny,)
    Nx, Ny = int(z_det["Nx"]), int(z_det["Ny"])

    # Reconstruct domain and spacings **from x,y**
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    Lx = float(x[-1] - x[0] + dx)
    Ly = float(y[-1] - y[0] + dy)

    # DO fields (physical, saved as real arrays)
    Ubar = z_do["U_save"]     # (Ny,Nx,Nt)
    Vbar = z_do["V_save"]
    Ui   = z_do["Ui_save"]    # (Ny,Nx,S,Nt)
    Vi   = z_do["Vi_save"]
    Ytms = z_do["YY_save"]    # (MC,S,Nt)
    CYY  = z_do["CYY_save"]   # (2,Nt): [∫(u^2+v^2)dxdy ; trace(C)]
    T_do = z_do["T_save"]
    dt   = float(z_do["dt"])
    Re   = float(z_do["Re"])

    S  = Ui.shape[2]
    MC = Ytms.shape[0]
    Nt = T_do.size
    assert Ubar.shape == (Ny, Nx, Nt)

    # ==== Wavenumbers consistent with (dx,dy) ====
    KX, KY = np.meshgrid(2*np.pi*fftfreq(Nx, d=dx),
                         2*np.pi*fftfreq(Ny, d=dy),
                         indexing='xy')

    # Parseval constant used by solver
    c_parseval = (Lx * Ly) / (Nx * Ny)**2

    # ==== Helpers ====
    def ip_scalar(a,b): return np.sum(a*b) * dx * dy
    def ip_vector(ux,uy,vx,vy): return ip_scalar(ux,vx) + ip_scalar(uy,vy)
    def l2(a): return np.sqrt(ip_scalar(a,a))

    def div_L2_from_phys(u, v):
        Uh, Vh = fft2(u), fft2(v)
        divh = 1j*KX*Uh + 1j*KY*Vh
        return np.sqrt(np.sum(np.abs(divh)**2) * c_parseval)

    def energy(u, v):  # kinetic energy = 0.5 ∫(u^2+v^2) dxdy
        return 0.5 * ip_scalar(u*u + v*v, 1.0)

    def time_deriv_centered(A, dt):
        dA = np.empty_like(A, dtype=float)
        dA[0]    = (A[1]    - A[0])    / dt
        dA[-1]   = (A[-1]   - A[-2])   / dt
        dA[1:-1] = (A[2:]   - A[:-2])  / (2*dt)
        return dA

    # ==== Time-first views ====
    Ubar_t = np.moveaxis(Ubar, -1, 0).real  # (Nt,Ny,Nx), real-only
    Vbar_t = np.moveaxis(Vbar, -1, 0).real
    Ui_ts  = np.moveaxis(Ui, (3,2,0,1), (0,1,2,3)).real  # (Nt,S,Ny,Nx)
    Vi_ts  = np.moveaxis(Vi, (3,2,0,1), (0,1,2,3)).real
    Ytms_t = np.moveaxis(Ytms, -1, 0)  # (Nt,MC,S)

    print(f"Grid {Nx}x{Ny}, Nt={Nt}, S={S}, MC={MC}, Re={Re}")

    # ==== 1) Orthonormality ====
    def gram_at_t(Ui_t, Vi_t):
        S_loc = Ui_t.shape[0]
        G = np.zeros((S_loc, S_loc))
        for i in range(S_loc):
            for j in range(S_loc):
                G[i,j] = ip_vector(Ui_t[i], Vi_t[i], Ui_t[j], Vi_t[j])
        return G

    G_err = np.zeros(Nt)
    for n in range(Nt):
        G = gram_at_t(Ui_ts[n], Vi_ts[n])
        G_err[n] = np.linalg.norm(G - np.eye(S), 'fro')
    print(f"[Orthonormality] max ||G-I||_F = {G_err.max():.3e}")

    # ==== 2) Incompressibility ====
    div_mean_max = 0.0
    div_mode_max = 0.0
    for n in range(Nt):
        div_mean_max = max(div_mean_max, div_L2_from_phys(Ubar_t[n], Vbar_t[n]))
        for i in range(S):
            div_mode_max = max(div_mode_max, div_L2_from_phys(Ui_ts[n,i], Vi_ts[n,i]))
    print(f"[Divergence] mean max ||∇·ū||={div_mean_max:.3e}; modes max ||∇·φ||={div_mode_max:.3e}")

    # ==== 3) Dynamic orthogonality ====
    Ui_dot = time_deriv_centered(Ui_ts, dt)
    Vi_dot = time_deriv_centered(Vi_ts, dt)
    do_viol = np.zeros(Nt)
    for n in range(Nt):
        M = np.zeros((S,S))
        for i in range(S):
            for j in range(S):
                M[i,j] = ip_vector(Ui_dot[n,i], Vi_dot[n,i], Ui_ts[n,j], Vi_ts[n,j])
        do_viol[n] = np.linalg.norm(M, 'fro')
    print(f"[Dyn. orthogonality] max ||〈φ̇,φ〉||_F = {do_viol.max():.3e}")

    # ==== 4) Energy budgets ====
    K_mean = np.array([energy(Ubar_t[n], Vbar_t[n]) for n in range(Nt)])

    # empirical trace(C) for any S
    trC_emp = np.zeros(Nt)
    for n in range(Nt):
        Y = Ytms_t[n]                           # (MC,S)
        Ym = Y - Y.mean(axis=0, keepdims=True)
        C  = (Ym.T @ Ym) / MC
        trC_emp[n] = np.trace(C)

    K_stoch_emp  = 0.5 * trC_emp
    K_total_emp  = K_mean + K_stoch_emp

    # From file: CYY[0] = ∫(u^2+v^2)dxdy  (no 1/2)
    K_mean_file  = CYY[0]                # make it comparable to K_mean
    var_file     = CYY[1]                      # trace(C)
    K_stoch_file = 0.5 * var_file
    K_total_file = K_mean_file + K_stoch_file

    print(f"[Energy fields]    K_mean(t0,tf)=({K_mean[0]:.6e},{K_mean[-1]:.6e}); "
          f"K_stoch=({K_stoch_emp[0]:.6e},{K_stoch_emp[-1]:.6e}); "
          f"K_total=({K_total_emp[0]:.6e},{K_total_emp[-1]:.6e})")
    print(f"[Energy from file] K_mean(t0,tf)=({K_mean_file[0]:.6e},{K_mean_file[-1]:.6e}); "
          f"K_stoch=({K_stoch_file[0]:.6e},{K_stoch_file[-1]:.6e}); "
          f"K_total=({K_total_file[0]:.6e},{K_total_file[-1]:.6e})")
    print(f"[Energy cross-check] "
          f"||K_mean(fields)-K_mean(file)||_∞={np.max(np.abs(K_mean-K_mean_file)):.3e}; "
          f"||var(emp)-var(file)||_∞={np.max(np.abs(trC_emp-var_file)):.3e}")

    # ==== 5) DO mean vs deterministic ====
    Udet = z_det["U_save"]; Vdet = z_det["V_save"]
    Udet_t = np.moveaxis(Udet, -1, 0).real
    Vdet_t = np.moveaxis(Vdet, -1, 0).real

    def match_times(Ta, Tb):  # indices in Tb nearest to times in Ta
        return np.array([np.argmin(np.abs(Tb - t)) for t in Ta])

    i_det = match_times(T_do, z_det["T_save"])
    err = np.zeros(Nt)
    for n in range(Nt):
        du = Ubar_t[n] - Udet_t[i_det[n]]
        dv = Vbar_t[n] - Vdet_t[i_det[n]]
        err[n] = np.sqrt(ip_scalar(du*du + dv*dv, 1.0))
    print(f"[DO mean vs Deterministic] L2 error: max={err.max():.3e}, median={np.median(err):.3e}")

    # ==== 6) Summary flags ====
    flags = {
        "orthonormality_ok": G_err.max() < 1e-8,
        "divergence_ok":     max(div_mean_max, div_mode_max) < 5e-4,
        "dyn_orth_ok":       do_viol.max() < 5e-6,
        "energy_match_ok":   (np.max(np.abs(K_mean-K_mean_file))<1e-8) and (np.max(np.abs(trC_emp-var_file))<1e-8),
        "do_vs_det_ok":      np.median(err) < 1e-3,
    }
    print("Summary flags:", flags)
def residual_norm(z_det, z_do, j_det, j_do, plot: bool = False) -> list:
    ubar = z_do["U_save"]; vbar = z_do["V_save"]
    udet = z_det["U_save"]; vdet = z_det["V_save"]

    Nx, Ny = int(z_do["Nx"]), int(z_do["Ny"])

    uv_do = np.vstack((np.reshape(ubar, shape=(Ny*Nx, -1)), np.reshape(vbar, shape=(Ny*Nx, -1))))
    uv_det = np.vstack((np.reshape(udet, shape=(Ny * Nx, -1)), np.reshape(vdet, shape=(Ny * Nx, -1))))

    l2_residual_norms = []

    for k in j_det:
        j = j_do[np.where(j_det == k)[0][0]]
        res_col = np.abs(uv_do[:, j] - uv_det[:, k])
        norm = np.linalg.norm(res_col)
        l2_residual_norms.append(norm)

    if plot:
        # plotting parameters
        plt.rcParams.update({
            "font.size": 17,
            "font.family": "serif",
            "text.usetex": True,
        })
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.set_facecolor((0.851, 0.912, 0.971, 0.8))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("L2 residual")
        ax.set_title("L2 residual of the velocity field, DO with S=0 vs deterministic.")
        ax.set_yscale("log")
        ax.grid()

        ax.plot(t_ref, l2_residual_norms, color="k")

        plt.show()

    return l2_residual_norms

if what == 'snapshots':
    z_det = np.load(DET_PATH, allow_pickle=True)
    z_do = np.load(DO_PATH, allow_pickle=True)
    dx, dy = z_do["Lx"] / z_do["Nx"], z_do["Ly"] / z_do["Ny"]

    x = np.linspace(0, z_do["Lx"] - dx, int(z_do["Nx"]))
    y = np.linspace(0, z_do["Ly"] - dy, int(z_do["Ny"]))
    xx, yy = np.meshgrid(x, y)

    udet, vdet = z_det["U_save"], z_det["V_save"]
    ubar, vbar = z_do["U_save"], z_do["V_save"]

    Nx, Ny = int(z_do["Nx"]), int(z_do["Ny"])
    t_do = np.asarray(z_do["T_save"], dtype=float)
    t_det = np.asarray(z_det["T_save"], dtype=float)
    T = ubar.shape[2]
    t_vals = z_do["T_save"]


    def nearest_indices(t_src, t_query):
        idx = np.searchsorted(t_src, t_query)
        idx0 = np.clip(idx - 1, 0, len(t_src) - 1)
        idx1 = np.clip(idx, 0, len(t_src) - 1)
        pick = np.where(np.abs(t_query - t_src[idx0]) <= np.abs(t_query - t_src[idx1]), idx0, idx1)
        return pick


    # Keep only DET times covered by DO range
    valid = (t_det >= t_do[0]) & (t_det <= t_do[-1])
    t_ref = t_det[valid]  # x-axis
    j_det = np.nonzero(valid)[0]  # indices in DET arrays
    j_do = nearest_indices(t_do, t_ref)  # matched indices in DO arrays

    res_norms = residual_norm(z_det, z_do, j_det, j_do)

    ts = [0.5, 2.5, 5, 7.5, 10]

    plt.rcParams.update({
        "font.size": 20,
        "font.family": "serif",
        "text.usetex": True,
    })
    fig = plt.figure(figsize=(22, 15), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    gs = GridSpec(3, len(ts), figure=fig, height_ratios=[5, 5, 2])

    cmap = plt.get_cmap("turbo")
    extent = [xx.min(), xx.max(), yy.min(), yy.max()]
    vmin = 0.0
    vmax = max(np.nanmax(np.hypot(udet, vdet)), np.nanmax(np.hypot(ubar, vbar))) - 1
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scale = 3
    s = slice(None, None, 4)
    for col, t in enumerate(ts):
        for row in range(2):
            ax = fig.add_subplot(gs[row, col])
            ax.set_aspect("equal")
            ax.set_xlabel("x", fontsize=20, fontfamily="serif")
            ax.set_ylabel("y", fontsize=20, fontfamily="serif")

            k = np.argwhere(t_det == t)[0][0]
            j = j_do[np.where(j_det == k)[0][0]]

            if row == 0:
                ax.set_title(rf"$\mathbf{{u}}_{{\mathrm{{det}}}}(\mathbf{{x}},t)$ at $t={t}$s", fontsize=20, fontfamily="serif")
                ut_det, vt_det = udet[..., k], vdet[..., k]
                speedt_det = np.hypot(ut_det, vt_det)
                imdet = ax.imshow(speedt_det, origin="lower", cmap=cmap, norm=norm, interpolation="bilinear",
                                  extent=extent)
                ax.quiver(xx[s, s], yy[s, s], ut_det[s, s], vt_det[s, s], color="k", angles="xy", scale_units="xy",
                          scale=scale,
                          units="inches", width=0.02, pivot="mid")
            else:
                ax.set_title(rf"$\mathbf{{u}}_{{\mathrm{{DO}}}}(\mathbf{{x}},t)$ at $t={t}$s", fontsize=20, fontfamily="serif")
                ut_do = ubar[..., j];
                vt_do = vbar[..., j]
                speedt_do = np.hypot(ut_do, vt_do)
                imdo = ax.imshow(speedt_do, origin="lower", cmap=cmap, norm=norm, interpolation="bilinear",
                                 extent=extent)
                ax.quiver(xx[s, s], yy[s, s], ut_do[s, s], vt_do[s, s], color="k", angles="xy", scale_units="xy",
                          scale=scale,
                          units="inches", width=0.02, pivot="mid")

    ax_res = fig.add_subplot(gs[2, :])
    ax_res.set_xlabel("t", fontsize=20, fontfamily="serif")
    ax_res.set_ylabel("Residual norm", fontsize=20, fontfamily="serif")
    ax_res.set_title(r"$\|\mathbf{u}_{det}-\mathbf{u}_{DO}\|$ timeseries")
    ax_res.plot(t_ref, res_norms, linewidth=0.5)
    ax_res.set_yscale("log")
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, location="right", fraction=0.035, pad=0.015)
    cbar.set_label(r"$|\mathbf{u}|$")
    fig.savefig("../save/DO_Lamb_Oseen/deterministic_comparison/N64_T10_S0_MC200/N-none/F-kolmogorov/plots/vel_evolution_norm_dtover20_detICs.pdf",dpi=300, bbox_inches="tight")
    plt.show()