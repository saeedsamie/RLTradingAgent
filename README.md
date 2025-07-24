# RLTradingAgent: XAUUSD Candlestick & Indicator Web Viewer

This project provides a pipeline for downloading, processing, and visualizing XAUUSD (Gold/USD) 5-minute candlestick data with technical indicators. It features a FastAPI backend and a modern web frontend using TradingView Lightweight Charts for interactive visualization.

## Features
- **Automated Data Download:** Fetches tick data from Alpari, aggregates to 5-minute OHLCV bars.
- **Technical Indicators:** Adds Moving Averages (MA20, MA50, MA200), RSI, and Ichimoku Cloud.
- **Data Cleaning:** Handles missing intervals and duplicate timestamps.
- **FastAPI Backend:** Serves processed data and a web UI.
- **Modern Chart UI:** Interactive candlestick chart with indicator overlays and tooltips, powered by TradingView Lightweight Charts.

## Requirements
- Python 3.8+
- pip

### Python Packages
- pandas
- numpy
- requests
- pandas_ta
- fastapi
- uvicorn
- jinja2
- orjson

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate and Process Data

Edit and run `dataset_generator.py` to:
- Download and aggregate raw data
- Fill missing intervals
- Add indicators
- Save to CSV/Parquet

Example (in `dataset_generator.py`):
```python
# Download, fill, and add indicators
# df5m = fetch_range_parallel(2015, 1, 2025, 7)
# df_filled = fill_time_series_holes(df5m, method='interpolate')
# df_ind = add_indicators(df_filled)
# df_ind.to_parquet("xauusd_5m_alpari.parquet")
```

### 2. Start the Web Server

```bash
uvicorn main_webview:app --reload
```

- The server will be available at [http://localhost:8000/](http://localhost:8000/)

### 3. View the Chart
- Open your browser and go to [http://localhost:8000/](http://localhost:8000/)
- You will see an interactive candlestick chart with MA20, MA50, MA200 overlays and RSI below.
- Hover your mouse to see indicator values in the top-left tooltip.

## File Structure
- `dataset_generator.py` — Data download, cleaning, and indicator pipeline
- `main_webview.py` — FastAPI backend and API
- `templates/tradingview.html` — Web UI (TradingView-style chart)
- `requirements.txt` — Python dependencies
- `xauusd_5m_alpari_filled_indicated.csv` — Example processed data file

## Credits
- [Alpari Tick Data](https://alpari.com/)
- [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts)
- [pandas_ta](https://github.com/twopirllc/pandas-ta)

## License
MIT License 