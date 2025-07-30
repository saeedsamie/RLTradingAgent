# RLTradingAgent: Advanced RL Trading System with Real-Time Dashboard

A comprehensive Reinforcement Learning (RL) trading system for XAUUSD (Gold/USD) with an advanced web-based training dashboard. Features automated data processing, RL agent training with PPO algorithm, and a modern real-time monitoring interface.

## 🚀 Key Features

### **Data Processing & Visualization**
- **Automated Data Pipeline:** Downloads tick data from Alpari, aggregates to 5-minute OHLCV bars
- **Technical Indicators:** Moving Averages (MA20, MA50, MA200), RSI, Ichimoku Cloud
- **Data Cleaning:** Handles missing intervals and duplicate timestamps
- **Interactive Charts:** Professional TradingView Lightweight Charts with floating tooltips

### **RL Training System**
- **PPO Algorithm:** Stable Baselines3 implementation for robust policy learning
- **Custom Trading Environment:** Gymnasium-based environment with realistic trading mechanics
- **Comprehensive Metrics:** Tracks rewards, loss, balance, equity, trades, PnL, learning rate, entropy
- **Real-Time Training Time:** Accurate training duration tracking

### **Advanced Training Dashboard**
- **Live Monitoring:** Real-time updates of all training metrics
- **Professional Charts:** 10 separate charts with TradingView Lightweight Charts
- **Interactive Tooltips:** Floating tooltips showing precise values on hover
- **Dynamic Controls:** Adjustable refresh intervals (10s, 30s, 60s, 2min, 5min)
- **Training Info Display:** Real training time, total timesteps, total episodes
- **High Precision:** Entropy chart with 10-decimal precision for tiny values

## 📊 Dashboard Charts

The training dashboard displays 10 comprehensive charts:

1. **Rewards Chart** - Episode rewards with mean reward tracking
2. **Lengths Chart** - Episode lengths with mean length tracking  
3. **Loss Chart** - Training loss progression
4. **Balance & Equity** - Account balance and equity over time
5. **Position Chart** - Current trading position (histogram)
6. **Total Trades** - Number of completed trades
7. **PnL Chart** - Total and unrealized profit/loss
8. **Learning Rate** - Policy learning rate (5-decimal precision)
9. **Entropy Chart** - Policy entropy (10-decimal precision for tiny values)

## 🛠️ Requirements

- Python 3.8+
- pip

### Python Packages
```
pandas numpy requests pandas_ta fastapi uvicorn jinja2 orjson
stable-baselines3 gymnasium
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. **Data Preparation**
```bash
# Generate and process data (if needed)
python dataset_generator.py
```

### 2. **Train RL Agent**
```bash
# Start training with live monitoring
python scripts/run_rl_pipeline.py
```

### 3. **Launch Web Dashboard**
```bash
# Start FastAPI server
uvicorn web.main_webview:app --reload
```

### 4. **Monitor Training**
Open [http://localhost:8000/training_dashboard](http://localhost:8000/training_dashboard) in your browser to see:
- ✅ **Live-updating charts** with 10-second refresh
- ✅ **Interactive tooltips** showing precise values
- ✅ **Training time display** with accurate duration
- ✅ **Dynamic refresh controls** (10s, 30s, 60s, 2min, 5min)
- ✅ **High-precision entropy tracking** (10 decimal places)

## 📁 Project Structure

```
RLTradingAgent/
├── RL/                          # RL training modules
│   ├── train_agent.py          # PPO training with metrics logging
│   ├── trading_env.py          # Custom Gym trading environment
│   └── data_prep.py           # Data loading and preprocessing
├── web/                        # FastAPI backend
│   └── main_webview.py        # API endpoints and server
├── templates/                  # Web interfaces
│   ├── training_dashboard.html # Advanced RL training dashboard
│   └── tradingview.html       # Interactive chart viewer
├── plots/                      # Training outputs
│   └── training_metrics.json  # Real-time training metrics
├── scripts/                    # Pipeline scripts
│   └── run_rl_pipeline.py    # Complete training pipeline
└── data/                      # Market data files
    └── xauusd_5m_alpari_filled_indicated.csv
```

## 🔧 Advanced Features

### **Real-Time Dashboard Features**
- **Floating Tooltips:** Professional tooltips that follow your mouse
- **Precision Control:** Learning rate (5 decimals), Entropy (10 decimals)
- **Boundary Detection:** Tooltips automatically hide when outside chart area
- **Timestamp Conversion:** Seamless conversion between timesteps and timestamps
- **Error Handling:** Graceful handling of missing or invalid data

### **Training Metrics**
- **Performance Tracking:** Rewards, loss, balance, equity, position
- **Trading Metrics:** Total trades, commission, PnL (realized/unrealized)
- **Learning Metrics:** Learning rate, entropy with high precision
- **Time Tracking:** Accurate training duration in seconds

### **Data Processing**
- **Missing Data Handling:** Interpolation and forward-fill methods
- **Technical Indicators:** Moving averages, RSI, Ichimoku cloud
- **Data Validation:** Duplicate removal and timestamp alignment

## 📈 API Endpoints

- `GET /` - Main chart viewer
- `GET /training_dashboard` - RL training dashboard
- `GET /api/training_metrics` - Raw training metrics JSON
- `GET /api/chart_data` - Processed chart data

## 🎯 Usage Examples

### **Monitor Training Progress**
```bash
# Start training
python scripts/run_rl_pipeline.py

# In another terminal, start dashboard
uvicorn web.main_webview:app --reload

# Open browser to http://localhost:8000/training_dashboard
```

### **View Market Data**
```bash
# Start server
uvicorn web.main_webview:app --reload

# Open browser to http://localhost:8000/
```

### **API Access**
```bash
# Get raw training metrics
curl http://localhost:8000/api/training_metrics

# Get chart data
curl http://localhost:8000/api/chart_data
```

## 🔍 Troubleshooting

### **Dashboard Not Updating**
- Check if training is running and generating `plots/training_metrics.json`
- Verify server is running on port 8000
- Check browser console for JavaScript errors

### **Charts Not Displaying**
- Ensure `training_metrics.json` contains valid data
- Check that all required fields are present in the JSON
- Verify TradingView Lightweight Charts library is loading

### **Tooltips Not Working**
- Ensure JavaScript is enabled in browser
- Check that chart containers are properly positioned
- Verify crosshair events are being triggered

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📚 Credits

- **[Alpari Tick Data](https://alpari.com/)** - Market data source
- **[TradingView Lightweight Charts](https://github.com/tradingview/lightweight-charts)** - Professional charting library
- **[Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)** - RL algorithms
- **[pandas_ta](https://github.com/twopirllc/pandas-ta)** - Technical indicators
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework

## 📄 License

MIT License - see LICENSE file for details.

---

**Ready to train your RL trading agent? Start with `python scripts/run_rl_pipeline.py` and monitor progress at `http://localhost:8000/training_dashboard`!** 