FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# netcdf4 needs the HDF5 system library at runtime.
# libhdf5-103 is outdated on newer Debian images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libhdf5-310 libhdf5-hl-310 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]