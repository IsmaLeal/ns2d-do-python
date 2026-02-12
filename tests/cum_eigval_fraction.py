import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--inputs", required=True, help="Comma-separated list of values of S.")
    ap.add_argument("--window_frac", type=float, default=0.65)
    ap.add_argument("--sigma", type=float, default = 500)

    args = ap.parse_args()

    # get S-values
    s_vals = [int(p.strip()) for p in args.inputs.split(",")]
    s_ref = np.max([int(s) for s in s_vals])

    lambdas = {}

    root = Path("./save/DO_Lamb_Oseen/runs/")
    for s in s_vals:
        keycheck = f"_S{s}_"
        matching_dir = [p for p in root.iterdir() if keycheck in p.name and p.is_dir()]
        if len(matching_dir) > 1:
            print(f"More than one directory for the value S = {s}.")
        data_path = matching_dir[0] / "N-spectral_ou" / "F-none" / "data" / f"fields_ouamp-{float(args.sigma)}.npz"
        data = np.load(data_path, allow_pickle=True)

        CYY = np.array(data["CYY_save"], dtype=float)

        eigvals = CYY[1:, :]
        eigvals_window = eigvals[:, int(np.floor(args.window_frac * eigvals.shape[1])):]
        avg_eigvals = np.mean(eigvals_window, axis=1)
        lambdas[s] = avg_eigvals

    denominator = np.sum(lambdas[s_ref])

    G = {}
    for s in s_vals:
        x = np.arange(1, s+1)
        y = np.cumsum(lambdas[s]) / denominator
        G[s] = (x, y)

    plt.figure()
    for s in s_vals:
        x, y = G[s]
        plt.plot(x, y, label=f"S={s}", lw=2, marker="o")
    plt.plot(np.arange(1, s_ref + 1), np.arange(1, s_ref+1)/s_ref, label=r"$y=x$ line")
    plt.xlabel("mode index i")
    plt.ylabel(r"$G_i^{(S)}$")
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.show()

    a = 1

if __name__=="__main__":
    main()