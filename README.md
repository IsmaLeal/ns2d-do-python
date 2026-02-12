# ns2d-do-python

Dynamically Orthogonal (DO) solver for 2D incompressible Navier-Stokes in a periodic box, with deterministic and stochastic spectral forcing, plus plotting and post-processing scripts.

## What this project does

- Solves 2D incompressible Navier-Stokes pseudo-spectrally.
- Uses a DO decomposition into:
  - Mean flow
  - `S` orthonormal stochastic modes
  - `MC` stochastic coefficients
- Supports forcing options:
  - Deterministic: `none`, `kick`, `kolmogorov`
  - Stochastic: Ornstein-Uhlenbeck in spectral space (`spectral_ou`)
- Saves compressed simulation outputs (`.npz`) for replay and analysis.
- Includes scripts for diagnostics, spectra, energy balance, and movies.

## Repository layout

```text
ns2d_do/
  main.py                     # CLI entry point
  solver.py                   # Time integration and DO evolution
  setup.py                    # Initial conditions + initial DO basis
  coloured_noise_forcing.py   # Spectral OU forcing
  spectral.py                 # Spectral operators and products
  grids.py                    # Grid + wavenumber helpers
  stats.py                    # DO/statistical helper routines
  plotting.py                 # Real-time plotting class
  params.py                   # Parameter dataclass

tests/
  plots.py
  replay_plotting.py
  energy_balance.py
  spectral_energy_movie.py
  ...                         # extra plotting/movie/forcing scripts
```

## Requirements

Core runtime:

- Python 3.10+
- `numpy`
- `matplotlib`

Used by analysis scripts in `tests/`:

- `tqdm`
- `pandas`

Optional:

- `ffmpeg` (required for video writers in movie scripts)

Install quickly:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy matplotlib tqdm pandas
```

## Running a simulation

From repo root:

```bash
python -m ns2d_do.main --Nx 64 --Ny 64 --Tf 20 --dt 1e-3 --S 10 --MC 200 --Re 40
```

Example with deterministic Kolmogorov forcing:

```bash
python -m ns2d_do.main --forcing kolmogorov --f_amp 1.0
```

Example with stochastic spectral OU forcing:

```bash
python -m ns2d_do.main \
  --forcing_stochastic spectral_ou \
  --ou_tau 1.0 --ou_var 1.0 --ou_amp 1.0 \
  --ou_k_min 2.5 --ou_k_max 8.5
```

Enable live plotting during run:

```bash
python -m ns2d_do.main --realtime
```

## Main CLI arguments

- Grid/domain: `--Lx --Ly --Nx --Ny`
- Time: `--Tf --dt`
- DO dimensions: `--S --MC`
- Physics: `--Re --forcing --f_amp`
- Stochastic forcing: `--forcing_stochastic --ou_tau --ou_var --ou_amp --ou_k_min --ou_k_max`
- Initialization randomness: `--seed --MR --smallvar --eps_IC_noise`
- IO/plot: `--outdir --PlotInterval --SaveInterval --plotIC --plot --realtime`
- Performance: `--threads`

See all options:

```bash
python -m ns2d_do.main -h
```

## Output format

By default, data is written under:

```text
save/DO_Lamb_Oseen/runs/N{Nx}_T{Tf}_S{S}_MC{MC}/N-{forcing_stochastic}/F-{forcing}/data/
```

Typical files:

- `fields_eps_<eps>.npz`
- `fields_smallICvar.npz`
- `fields_ouamp-<amp>.npz`

Saved arrays include simulation parameters and time histories such as:

- `T_save`, `U_save`, `V_save`, `Ui_save`, `Vi_save`
- `W_save`, `Wi_save`, `YY_save`
- `CYY_save`, `CYY_plot`, `T_plot`
- forcing metadata (`ou_*`, `half_k_pairs`, `ou_amplitudes`)

## Post-processing scripts

Examples:

```bash
python tests/plots.py --help
python tests/replay_plotting.py --help
python tests/energy_balance.py --help
python tests/spectral_energy_movie.py --help
```

Note: movie scripts require a working `ffmpeg` installation.

## Reproducibility

- Use `--seed` to control stochastic initialization/forcing reproducibly.
- Keep `--Nx`, `--Ny`, `--dt`, `--S`, `--MC`, and forcing settings fixed across runs for fair comparisons.

## License

This project is distributed under the terms in `LICENSE`.
