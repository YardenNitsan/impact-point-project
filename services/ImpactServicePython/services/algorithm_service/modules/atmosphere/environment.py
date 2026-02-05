import requests

CELSIUS_TO_KELVIN = 273.15
HPA_TO_PA = 100.0


def get_sea_level_environment(lat: float, lon: float):
    """
    Returns sea-level environmental conditions in SI units:
    T0 [K], P0 [Pa]
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": True,
        "hourly": "surface_pressure"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    T0_celsius = data["current_weather"]["temperature"]
    P0_hpa = data["hourly"]["surface_pressure"][0]

    T0 = T0_celsius + CELSIUS_TO_KELVIN
    P0 = P0_hpa * HPA_TO_PA

    return T0, P0
