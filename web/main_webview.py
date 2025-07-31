import json
import os
import logging

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Static files are not needed as the template uses CDN for libraries and inline CSS

# Path to your filled and indicated CSV
DATA_PATH = "xauusd_5m_alpari_filled_indicated.csv"


@app.get("/chart", response_class=HTMLResponse)
def tradingview_widget(request: Request):
    """
    Serves the TradingView widget page.
    """
    return templates.TemplateResponse("tradingview.html", {"request": request})


@app.get("/", response_class=HTMLResponse)
def training_dashboard(request: Request):
    """
    Serves the RL training dashboard page.
    """
    return templates.TemplateResponse("training_dashboard.html", {"request": request})


@app.get("/api/candles", response_class=ORJSONResponse)
def get_candles():
    """
    Returns OHLCV and indicator data as JSON (for custom overlays if needed).
    """
    if not os.path.exists(DATA_PATH):
        return {"error": "Data file not found."}
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.reset_index()
    # Only keep the last 1000 rows
    df = df.tail(1000)
    # Convert Timestamp to ISO string
    df['index'] = df['index'].astype(str)
    # Replace inf, -inf with None
    df = df.replace([float('inf'), float('-inf')], None)
    # Replace NaN/NA with None
    df = df.where(pd.notnull(df), None)
    data = df.to_dict(orient="records")
    return data


@app.get("/api/training_metrics", response_class=ORJSONResponse)
def get_training_metrics():
    """
    Returns the latest training metrics as JSON for live or on-demand plotting.
    """
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots', 'training_metrics.json')
    if not os.path.exists(metrics_path):
        return {"error": "No training metrics found."}
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics


@app.get("/api/training_config", response_class=ORJSONResponse)
def get_training_config():
    """
    Returns the current training configuration including total timesteps.
    """
    try:    
        from scripts.run_rl_pipeline import config
        logging.info(config)
        return {
            "total_timesteps": config['total_timesteps'],
            "description": config['description'],
            "window_size": config['window_size'],
            "max_episode_steps": config['max_episode_steps']
        }
    except Exception as e:
        return {"error": f"Failed to load config: {str(e)}"}
