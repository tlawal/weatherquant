FROM python:3.12-slim

# Install system deps:
#   lxml (BeautifulSoup WU scraping) — libxml2-dev libxslt-dev
#   netCDF4 (Open-Meteo decoding) — libhdf5-dev libnetcdf-dev
#   cfgrib / herbie GRIB2 decoding — libeccodes-dev
RUN apt-get update && apt-get install -y \
    libxml2-dev libxslt-dev gcc libhdf5-dev libnetcdf-dev libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TZ=America/New_York
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# SERVICE_TYPE is set by Railway environment variable (api | worker)
CMD ["python", "-m", "backend.main"]
