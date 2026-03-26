from dataclasses import asdict, dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class BasisConfig:
    degree: int = 3

    # Main spatial smooth.
    lat_basis_space: int = 24
    lon_basis_space: int = 48

    # Main 1D smooths.
    alt_basis: int = 24
    day_basis: int = 40

    # Interaction bases.
    alt_basis_interaction: int = 12
    lat_basis_interaction: int = 12
    lon_basis_interaction: int = 24
    day_basis_interaction: int = 12

    # Harmonics.
    local_hour_harmonics: int = 6
    utc_hour_harmonics: int = 3

    lat_range: Tuple[float, float] = (-90.0, 90.0)
    lon_range: Tuple[float, float] = (-180.0, 180.0)
    alt_range_m: Tuple[float, float] = (0.0, 32000.0)
    day_of_year_range: Tuple[float, float] = (1.0, 366.0)

    # Ridge strengths. The model is intentionally richer than the original,
    # so the interactions are regularized more strongly than the marginal terms.
    ridge_space: float = 2e-2
    ridge_alt: float = 1e-2
    ridge_day: float = 2e-2
    ridge_interaction: float = 8e-2
    ridge_time_harmonics: float = 2e-3

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class SamplingConfig:
    samples_per_file: int = 12000
    seed: int = 42
    altitude_clip_m: Tuple[float, float] = (0.0, 32000.0)
    stratified_time_level: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class SplitConfig:
    train_end_day_inclusive: int = 23
    val_end_day_inclusive: int = 27

    def to_dict(self) -> Dict:
        return asdict(self)