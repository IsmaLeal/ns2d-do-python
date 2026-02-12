from ns2d_do.coloured_noise_forcing import OU, SpectralOUForcing
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def acf(x: np.ndarray, max_lag: int, *, demean: bool = True, biased: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (lags, normalised autocorrelation function (ACF)) for lags between
    0 and max_lag-1.
    `biased=True` uses 1/N normalisation; `False` uses 1/(N-lag).
    """
    if demean: x = x - x.mean()
    N = x.size
    lags = np.arange(max_lag, dtype=int)
    denom = N if biased else (N - lags)
    out = np.empty(max_lag, dtype=float)
    for k in lags:
        out[k] = np.dot(x[:N - k], x[k:])
    out /= denom
    out = out / out[0]
    return lags, out


plt.rcParams.update({'font.size': 15, 'lines.linewidth': 0.5})

# parameters
dt = 0.001
mu = 0
tau = 10
var = 25
tau2 = 1
var2 = 25
tau3 = 10
var3 = 5
amp = 4
N_samples = int(5e5)
t = np.arange(0, N_samples) * dt

# random number generators
rng = np.random.default_rng(1)
rng2 = np.random.default_rng(2)
rng3 = np.random.default_rng(3)

# largest burn-in
burnin_length = int((10 * max(tau, tau2, tau3)) / dt)

# OU processes
ou = OU(dt, mu, tau, var, rng)
ou2 = OU(dt, mu, tau2, var2, rng)
ou3 = OU(dt, mu, tau3, var3, rng)

# draw samples
samples1 = np.empty(N_samples)
samples2 = np.empty(N_samples)
samples3 = np.empty(N_samples)
for i in tqdm(range(N_samples), desc="Simulating OU processes"):
    samples1[i] = ou.step()
    samples2[i] = ou2.step()
    samples3[i] = ou3.step()

# plot full time series
fig, axs = plt.subplots(2, 1, figsize=(15, 11))
plt.subplots_adjust(hspace=0.3)

ax = axs[0]
ax.plot(t, samples1, label=r"$\tau=10,\mathrm{var}=25$")
ax.plot(t, samples2, label=r"$\tau=1,\mathrm{var}=25$")
ax.plot(t, samples3, label=r"$\tau=10,\mathrm{var}=5$")
ax.set_xlabel("Time (s)")
ax.set_ylabel("OU samples")
ax.set_title("OU samples vs time")
ax.legend()

# Remove burn-in samples1
x1 = samples1[burnin_length:]
x2 = samples2[burnin_length:]
N_kept = x1.size

max_lag = int(5 *tau / dt)
lags, R1 = acf(x1, max_lag)
_, R2 = acf(x2, max_lag)
lag_tau = lags * dt

ax = axs[1]
ax.plot(lags*dt, R1, label=r"ACF, $\tau=10$", linewidth=3)
ax.plot(lags*dt, np.exp(-np.abs(lags*dt) / tau), '--', label=r"Expected ACF, $\tau=10$", linewidth=3)
ax.plot(lags*dt, R2, label=r"ACF, $\tau=1$", linewidth=3)
ax.plot(lags*dt, np.exp(-np.abs(lags*dt) / tau2), '--', label=r"Expected ACF, $\tau=1$", linewidth=3)
ax.set_xlabel("Lag (s)")
ax.set_ylabel("Autocorrelation")
ax.set_title(r"Autocorrelation examples, considering lags up to $5\tau$ and burn-in period of $10\tau$")
ax.legend()

plt.show()