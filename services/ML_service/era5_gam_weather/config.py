from dataclasses import asdict, dataclass
from typing import Dict, Tuple


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