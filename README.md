# RLTradingAgent: XAUUSD Candlestick & Indicator Web Viewer

This project provides a pipeline for downloading, processing, and visualizing XAUUSD (Gold/USD) 5-minute candlestick data with technical indicators. It features a FastAPI backend and a modern web frontend using TradingView Lightweight Charts for interactive visualization.

## Features
- **Automated Data Download:** Fetches tick data from Alpari, aggregates to 5-minute OHLCV bars.
- **Technical Indicators:** Adds Moving Averages (MA20, MA50, MA200), RSI, and Ichimoku Cloud.
- **Data Cleaning:** Handles missing intervals and duplicate timestamps.
- **FastAPI Backend:** Serves processed data and a web UI.
- **Modern Chart UI:** Interactive candlestick chart with indicator overlays and tooltips, powered by TradingView Lightweight Charts.
- **RL Training Dashboard:** Live web dashboard for monitoring RL agent training metrics in real time (see below).

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
- stable-baselines3
- gymnasium

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

### 2. Train the RL Agent and Monitor Training

Start a training run (e.g., using the pipeline script):
```bash
python scripts/run_rl_pipeline.py
```
This will train the RL agent and log all training metrics to `plots/training_metrics.json`.

#### Start the FastAPI Web Server
In a separate terminal, run:
```bash
uvicorn web.main_webview:app --reload
```

#### Open the Training Dashboard
Go to [http://localhost:8000/training_dashboard](http://localhost:8000/training_dashboard) in your browser.
- The dashboard will show live-updating plots of all training metrics, including:
  - Episode reward, mean reward (100)
  - Episode length, mean length (100)
  - Loss
  - Balance, equity, position
  - Total trades, total commission, total/unrealized PnL
  - Price, commission
  - Learning rate
  - Entropy
- The dashboard auto-refreshes every 10 seconds.

#### Test the API Directly
You can also view the raw metrics at [http://localhost:8000/api/training_metrics](http://localhost:8000/api/training_metrics)

### 3. View the Chart
- Open your browser and go to [http://localhost:8000/](http://localhost:8000/)
- You will see an interactive candlestick chart with MA20, MA50, MA200 overlays and RSI below.
- Hover your mouse to see indicator values in the top-left tooltip.

## File Structure
- `dataset_generator.py` — Data download, cleaning, and indicator pipeline
- `main_webview.py` — FastAPI backend and API
- `templates/tradingview.html` — Web UI (TradingView-style chart)
- `templates/training_dashboard.html` — RL training dashboard (live metrics)
- `requirements.txt` — Python dependencies
- `xauusd_5m_alpari_filled_indicated.csv` — Example processed data file

## RL Trading Modules

- `data_prep.py`: Data loading and missing interval checking
- `trading_env.py`: Custom Gym trading environment
- `train_agent.py`: RL agent training logic (with metrics logging)
- `evaluate.py`: Evaluation and metrics
- `plotting.py`: Plotting utilities
- `run_rl_pipeline.py`: Main script to run the full pipeline

## Credits
- [Alpari Tick Data](https://alpari.com/)
- [TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts)
- [pandas_ta](https://github.com/twopirllc/pandas-ta)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)

## License
MIT License 