import matplotlib.pyplot as plt
import numpy as np

dts = np.array([1e-3, 5e-4, 1e-4, 5e-5])
res = np.array([0.3037838, 0.142538, 0.02237, 0.012081])

fig, ax = plt.subplots(figsize=(4.5,3.5))
ax.xaxis.set_inverted(True)
ax.loglog(dts, res, marker="o", lw=1, ms=5, label=r"Empirical ")
ax.set_xlabel(r"DO timestep $\mathrm{d}t$")
ax.set_ylabel(r"$E_{\infty}=\max_t \| \mathbf{u}_{DO} - \mathbf{u}_{det}\|$")
ax.grid(True, alpha=0.3)
ax.tick_params(direction="in", which="both")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- reference slope line (pure O(dt)) ---
x0, y0 = 5e-5, 1e-2      # anchor
x_ref = np.array([x0, x0*20])
y_ref = y0 * (x_ref / x0)**1   # slope = 1 in log–log

ax.loglog(x_ref, y_ref, "k--", lw=1.0, label=r"$\mathcal{O}(\mathrm{d}t)$")

ax.legend(frameon=False)
fig.tight_layout()
plt.show()