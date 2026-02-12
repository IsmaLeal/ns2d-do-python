from __future__ import annotations
from typing import Optional, Any, Dict
from dataclasses import dataclass
import numpy as np
import pathlib

try:
    import yaml
except Exception:
    yaml = None

@dataclass
class Params:
    # domain/grid
    Lx: float = 2.0 * np.pi
    Ly: float = 2.0 * np.pi
    Nx: int = 64
    Ny: int = 64
    # time
    Tf: float = 3.0
    dt: float = 1e-3
    # DO subspace & sampling
    S: int = 6
    MC: int = 200
    # physical parameters
    Re: float = 40.0
    # options
    PlotInterval: int = 10
    SaveInterval: int = 10
    # initialisation
    eps_IC_noise: float = 0.09
    forcing: str = "none"
    f_amp: float = 1.0
    forcing_stochastic: str = "none"
    ou_tau: float = 2.0
    ou_var: float = 1.0
    ou_amp: float = 1.0
    ou_k_min: float = 3.5
    ou_k_max: float = 4.5
    Seed: Optional[int] = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @staticmethod
    def from_yaml(path: str | pathlib.Path) -> "Params":
        if yaml is None:
            raise RuntimeError("Install pyyaml to load Params from YAML.")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return Params(**data)
