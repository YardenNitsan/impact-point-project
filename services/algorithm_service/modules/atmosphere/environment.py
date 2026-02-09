import requests

CELSIUS_TO_KELVIN = 273.15
HPA_TO_PA = 100.0

# ISA sea-level defaults (robust fallback)
ISA_T0 = 288.15      # [K]
ISA_P0 = 101325.0    # [Pa]


def get_sea_level_environment(lat: float, lon: float):
    """Return sea-level environmental conditions in SI units: T0 [K], P0 [Pa].

    Uses Open-Meteo when available; falls back to ISA sea-level if the API is unreachable.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "surface_pressure",
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        T0_celsius = data["current_weather"]["temperature"]
        P0_hpa = data["hourly"]["surface_pressure"][0]

        T0 = float(T0_celsius) + CELSIUS_TO_KELVIN
        P0 = float(P0_hpa) * HPA_TO_PA

        # basic sanity
        if not (150.0 <= T0 <= 350.0 and 5e4 <= P0 <= 1.2e5):
            return ISA_T0, ISA_P0

        return T0, P0

    except Exception:
        return ISA_T0, ISA_P0
