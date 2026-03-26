import calendar
from pathlib import Path

import cdsapi

c = cdsapi.Client()

YEAR = "2025"

# Change this to the months you want.
# If you only want the missing two months back, keep 03 and 04.
MONTHS = ["03", "04"]

PRESSURE_LEVELS = [
    "1000", "925", "850", "700", "600",
    "500", "400", "300", "250", "200",
    "150", "100", "70", "50",
    "30", "20", "10"
]

TIMES = [
    "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
    "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
    "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
]

# Script is in: project_root/scripts/download_era5_2025.py
# This points to: project_root/data/era5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "era5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for month in MONTHS:
    num_days = calendar.monthrange(int(YEAR), int(month))[1]

    for day_num in range(1, num_days + 1):
        day = f"{day_num:02d}"
        file_path = OUTPUT_DIR / f"era5_{YEAR}_{month}_{day}.nc"

        if file_path.exists():
            print(f"Skipping {YEAR}-{month}-{day} (already exists)")
            continue

        print(f"Downloading {YEAR}-{month}-{day} -> {file_path}")

        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "temperature",
                    "u_component_of_wind",
                    "v_component_of_wind",
                    "geopotential",
                ],
                "pressure_level": PRESSURE_LEVELS,
                "year": YEAR,
                "month": month,
                "day": day,
                "time": TIMES,
                "format": "netcdf",
            },
            str(file_path),
        )

print("All downloads finished.")