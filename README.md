# WeatherQuant

Quantitative prediction market trading system for weather events on Polymarket.

## External Data Sources

### Weather Underground (WU)
WU serves as the official settlement resolution source for many Polymarket weather markets. To prevent staleness, the system scrapes the following URLs automatically every 5 minutes:
- **Daily Forecast/Summary**: `https://www.wunderground.com/weather/{METAR}`
- **Hourly Projections**: `https://www.wunderground.com/hourly/{METAR}`

*(Example: https://www.wunderground.com/weather/KATL)*

### METAR (Aviation Weather)
Provides the official real-time temperature observations from airport stations (ground truth).
- **Latest Observation**: `https://aviationweather.gov/api/data/metar?ids={METAR}&format=json&latest=1`

### National Weather Service (NWS)
Provides highly reliable baseline gridpoint forecasts.
