from era5_gam_weather.model import WeatherGAMBundle

MODEL_PATH = "artifacts/weather_model_bundle_2025_05.npz"


def main() -> None:
    model = WeatherGAMBundle.load(MODEL_PATH)
    pred = model.predict_one(
        lat=31.7683,
        lon=35.2137,
        altitude_m=22000.0,
        day_of_year=135.0,
        utc_hour=12.0,
    )
    print(pred)


if __name__ == "__main__":
    main()