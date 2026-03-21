FROM python:3.12-slim

# Install system deps (lxml for BeautifulSoup WU scraping)
RUN apt-get update && apt-get install -y \
    libxml2-dev libxslt-dev gcc \
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
