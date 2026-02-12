import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = "../save/DO_Lamb_Oseen/runs/N64_T20_S10_MC200/N-spectral_ou/F-none/data/fields_ouamp-1000.0.npz"
npz_path = Path(path)
out_path = npz_path.parents[1] / "plots" / "energies_sigma1000.pdf"         # e.g. .../F-none/plots/energies.pdf
out_path.parent.mkdir(parents=True, exist_ok=True)

data = np.load(npz_path)
CYY = data["CYY_save"]      # shape (1+S, nt)
t = data["T_save"]

# ---- plot ----
plt.rcParams.update({"font.size": 20})
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.set_title(r"Energy (mean, modes)")
ax.set_xlabel("Time")
ax.set_xlim(t[0], t[-1])
ax.tick_params(axis="both", which="both", labelsize=16)

ax.plot(t, CYY[0, :],  label="Mean",        ls="--", color="k")
for i in range(1, CYY.shape[0]):
    ax.plot(t, CYY[i, :],  label=f"Mode {i}", ls="-")
ax.legend(
    loc="upper right",
    fontsize=12,
    frameon=True,
    framealpha=0.9,
    borderpad=0.3,
    labelspacing=0.3,
    handlelength=1.6,
    handletextpad=0.5,
)

fig.savefig(out_path, bbox_inches="tight")
