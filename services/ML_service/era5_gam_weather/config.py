from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class SamplingConfig:
    samples_per_file: int = 12000
    seed: int = 42
    altitude_clip_m: Tuple[float, float] = (0.0, 32000.0)
    stratified_time_level: bool = True
    # When True, sample lat indices proportional to cos(lat) so polar grid
    # cells (which are tiny in physical area) are not over-represented.
    area_weighted_lat: bool = True
    # Optional per-pressure-level sample weights. If provided, len(level_weights)
    # must equal n_level for each ERA5 file; the per-level sample budget is
    # multiplied by these weights (and renormalized to samples_per_file). Use
    # this to emphasize lower-troposphere or jet-level samples.
    level_weights: Optional[Tuple[float, ...]] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class SplitConfig:
    train_end_day_inclusive: int = 23
    val_end_day_inclusive: int = 27

    def to_dict(self) -> Dict:
        return asdict(self)