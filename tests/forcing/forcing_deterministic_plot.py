import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LogNorm

# parameters
Nx, Ny = 64, 64
Lx, Ly = 2 * np.pi, 2 * np.pi
dx, dy = Lx / Nx, Ly / Ny
xx, yy = np.meshgrid(np.arange(0, Lx, dx), np.arange(0, Ly, dy))
forcing = "kolmogorov"

# plotting parameters
plt.rcParams.update({
    "font.size": 16,
    "font.family": "serif",
    "text.usetex": True,
})
arrwidth = 0.02; scale = 3; stride = 3
s = slice(None, None, stride)
xs, ys = xx[s, s], yy[s, s]
cmap = plt.get_cmap("coolwarm") if "coolwarm" in plt.colormaps() else plt.get_cmap("seismic")
fig, ax = plt.subplots(1, 1, figsize=(9, 6), constrained_layout=True)
ax.set_facecolor((0.851, 0.912, 0.971, 0.8))

# build forcing arrays
if forcing == "kolmogorov":
    Fx = np.sin(4 * yy)
    Fy = np.zeros_like(Fx)

    cf = ax.contourf(xx, yy, Fx, levels=1000, cmap="turbo", alpha=0.9)
    cblabel = r"$f_x$"
    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cblabel)
elif forcing == "kick":
    Fx = np.exp(-4 * (xx - Lx / 2)**2 - 4 * (yy - Ly / 2)**2) * (2 + np.tanh(yy - Ly / 2))
    Fy = np.zeros_like(Fx)

    cf = ax.contourf(xx, yy, Fx, levels=1000, cmap="turbo", alpha=0.9)
    cblabel = r"$f_x$"
    cbar = plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cblabel)
else:
    print("Coercing forcing to none.")
    Fx = np.zeros((Ny, Nx), dtype=float)
    Fy = np.zeros((Ny, Nx), dtype=float)

# plot
# ax.set_facecolor("#d9e9f9")
ax.grid(color="gray", linestyle='-', linewidth=0.7)
ax.set_aspect("equal")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_xlim(0, Lx-dx); ax.set_ylim(0, Ly-dy)
ax.set_title(f"Deterministic, steady forcing: {forcing}")
ax.tick_params(direction="in", length=4, width=0.8)

ax.quiver(
    xs, ys, Fx[s, s], Fy[s, s],
    color="k",
    width=arrwidth, angles="xy",
    scale_units="xy", scale=scale, units="inches",
    pivot="mid", headwidth=3
)
plt.show()