"""Weather ML package.

The package init stays lightweight so the new TensorFlow/Keras backend can be
imported without accidentally importing legacy backend code.
"""

from .config import SamplingConfig, SplitConfig
from .weather_features import WeatherFeatureBuilder, WeatherFeatureConfig

__all__ = [
    "SamplingConfig",
    "SplitConfig",
    "WeatherFeatureBuilder",
    "WeatherFeatureConfig",
    "MultiHeadMLPWeatherModel",
    "WeatherTreeBundle",
    "WeatherTreeTrainer",
    "NumpyMLPWeatherModel",
]


def __getattr__(name: str):
    if name == "MultiHeadMLPWeatherModel":
        from .multi_head_mlp_model import MultiHeadMLPWeatherModel
        return MultiHeadMLPWeatherModel
    if name in {"WeatherTreeBundle", "WeatherTreeTrainer"}:
        from .tree_model import WeatherTreeBundle, WeatherTreeTrainer
        return {"WeatherTreeBundle": WeatherTreeBundle, "WeatherTreeTrainer": WeatherTreeTrainer}[name]
    if name == "NumpyMLPWeatherModel":
        from .numpy_mlp_model import NumpyMLPWeatherModel
        return NumpyMLPWeatherModel
    raise AttributeError(name)
