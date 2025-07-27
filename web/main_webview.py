import os

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json

app = FastAPI()

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Serve static files (if needed for custom JS/CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Path to your filled and indicated CSV
DATA_PATH = "xauusd_5m_alpari_filled_indicated.csv"


@app.get("/", response_class=HTMLResponse)
def tradingview_widget(request: Request):
    """
    Serves the TradingView widget page.
    """
    return templates.TemplateResponse("tradingview.html", {"request": request})


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
    # df = df.tail(1000)
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
