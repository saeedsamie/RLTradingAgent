# RLTradingAgent: Advanced RL Trading System with Market Cycle Optimization

A comprehensive Reinforcement Learning (RL) trading system for XAUUSD (Gold/USD) with **market cycle-based configurations** and an advanced web-based training dashboard. Features automated data processing, optimized RL agent training with quarterly episodes, and a modern real-time monitoring interface.

## 🚀 Key Features

### **Market Cycle Optimization** ⭐ **ENHANCED**
- **Quarterly Episodes**: 25,920 bars (90 days) for comprehensive market cycle learning
- **Daily Windows**: 288 bars (1 day) for optimal pattern recognition
- **Configurable Cycles**: Easy switching between hourly, daily, weekly, monthly, quarterly, and yearly cycles
- **Optimized Timesteps**: 52.8M timesteps (60 epochs) for thorough learning
- **Market-Aware Training**: Episodes match natural market cycles and patterns
- **Multiple Configurations**: 8 different training configurations for various trading strategies

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
- **Market Cycle Learning:** Episodes designed to capture complete market cycles

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

## 🎯 Market Cycle Configurations

### **Available Configurations** ⭐ **EXPANDED**
```python
# Short-term trading (2M timesteps)
config = get_config('short_term')      # Hourly window, daily episodes

# Balanced approach (5M timesteps)  
config = get_config('medium_term')     # Daily window, weekly episodes

# Trend following (10M timesteps)
config = get_config('long_term')       # Weekly window, monthly episodes

# Long-term trends (20M timesteps)
config = get_config('trend_following') # Monthly window, quarterly episodes

# Optimal learning (52.8M timesteps) ⭐ RECOMMENDED
config = get_config('quarterly_focused') # Daily window, quarterly episodes

# Active trading (5M timesteps) ⭐ NEW
config = get_config('active_trading')  # Daily window, weekly episodes

# Improved trading (5M timesteps) ⭐ NEW
config = get_config('improved_trading') # Enhanced reward function

# Ultra aggressive (3M timesteps) ⭐ NEW
config = get_config('ultra_aggressive') # Forces trading activity

# Deep network (5M timesteps) ⭐ NEW
config = get_config('deep_network')    # Complex pattern learning
```

### **Configuration Details**
| Config | Window | Episode | Timesteps | Description |
|--------|--------|---------|-----------|-------------|
| `short_term` | 12 bars (1h) | 288 bars (1d) | 2M | Short-term trading |
| `medium_term` | 288 bars (1d) | 2016 bars (7d) | 5M | Balanced approach |
| `long_term` | 2016 bars (7d) | 8767 bars (30d) | 10M | Trend following |
| `trend_following` | 8767 bars (30d) | 25920 bars (90d) | 20M | Long-term trends |
| `quarterly_focused` | 288 bars (1d) | 25920 bars (90d) | 52.8M | **Optimal learning** |
| `active_trading` | 288 bars (1d) | 2016 bars (7d) | 5M | Active trading |
| `improved_trading` | 288 bars (1d) | 2016 bars (7d) | 5M | Enhanced rewards |
| `ultra_aggressive` | 288 bars (1d) | 2016 bars (7d) | 3M | Force trading |
| `deep_network` | 288 bars (1d) | 2016 bars (7d) | 5M | Complex patterns |

### **Why Quarterly Episodes Work Better**
- **Complete Market Cycles**: 90-day episodes capture full quarterly patterns
- **Seasonal Patterns**: Learns from earnings cycles and economic data
- **Trend Recognition**: Better understanding of long-term market movements
- **Realistic Trading**: Matches typical investment horizons
- **Reduced Noise**: Filters out short-term market fluctuations

## 🛠️ Requirements

- Python 3.8+
- pip

### Python Packages
```
pandas numpy requests pandas_ta fastapi uvicorn jinja2 orjson
stable-baselines3 gymnasium matplotlib tensorboard
torch torchvision torchaudio (with CUDA support)
pytest pytest-cov pytest-mock pytest-watch
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
# Start training with live monitoring (automatically resumes from latest checkpoint)
python scripts/run_rl_pipeline.py

# Start training from scratch (ignores existing checkpoints)
python scripts/run_rl_pipeline.py --fresh-start

# CPU-only training (skip GPU acceleration)
python scripts/run_rl_pipeline.py --cpu-only

# Debug mode with detailed logging
python scripts/run_rl_pipeline.py --debug

# Combine options
python scripts/run_rl_pipeline.py --fresh-start --cpu-only --debug
```

### 3. **Launch Web Dashboard**
```bash
# Start FastAPI server
uvicorn src.web.main:app --reload
```

### 4. **Monitor Training**
Open [http://localhost:8000/](http://localhost:8000/) in your browser to see:
- ✅ **Live-updating charts** with 10-second refresh
- ✅ **Interactive tooltips** showing precise values
- ✅ **Training time display** with accurate duration
- ✅ **Dynamic refresh controls** (10s, 30s, 60s, 2min, 5min)
- ✅ **High-precision entropy tracking** (10 decimal places)

## 📁 Project Structure

```
RLTradingAgent/
├── src/
│   ├── RL/                          # RL training modules
│   │   ├── train_agent.py          # PPO training with metrics logging
│   │   ├── trading_env.py          # Custom Gym trading environment
│   │   ├── evaluate.py             # Agent evaluation functions
│   │   └── plotting.py             # Advanced plotting utilities
│   ├── web/                        # FastAPI backend
│   │   └── main.py                # API endpoints and server
│   ├── templates/                  # Web interfaces
│   │   ├── training_dashboard.html # Advanced RL training dashboard
│   │   └── tradingview.html       # Interactive chart viewer
│   ├── scripts/                    # Pipeline scripts
│   │   ├── run_rl_pipeline.py    # Complete training pipeline
│   │   ├── data_prep.py          # Data preparation utilities
│   │   ├── manage_checkpoints.py # Checkpoint management
│   │   └── calculate_timesteps.py # Optimal timesteps calculator
│   └── config/                     # Configuration management
│       └── config.py              # Market cycle configurations
├── outputs/                        # Training outputs
│   ├── models/                     # Trained models and checkpoints
│   ├── plots/                      # Training plots and metrics
│   └── trade_history/             # Trading history logs
├── data/                          # Market data files
│   ├── raw/                       # Raw tick data
│   └── processed/                 # Processed OHLCV data
├── logs/                          # TensorBoard logs
├── tests/                         # Unit tests
└── requirements.txt               # Python dependencies
```

## 🔧 Advanced Features

### **Market Cycle Optimization** ⭐ **ENHANCED**
- **Quarterly Episodes**: 25,920 bars (90 days) for comprehensive learning
- **Daily Windows**: 288 bars (1 day) for optimal pattern recognition
- **Configurable Timesteps**: 52.8M timesteps (60 epochs) for thorough training
- **Market-Aware Training**: Episodes designed to capture complete market cycles
- **Easy Configuration**: Switch between different cycle lengths with one line
- **8 Training Configurations**: From ultra-aggressive to deep network learning

### **Command Line Options** ⭐ **NEW**
```bash
# Fresh start (clear all checkpoints)
python scripts/run_rl_pipeline.py --fresh-start

# CPU-only training (skip GPU)
python scripts/run_rl_pipeline.py --cpu-only

# Debug mode with detailed logging
python scripts/run_rl_pipeline.py --debug

# Combine options
python scripts/run_rl_pipeline.py --fresh-start --cpu-only --debug

# Show help
python scripts/run_rl_pipeline.py --help
```

### **Real-Time Dashboard Features**
- **Floating Tooltips:** Professional tooltips that follow your mouse
- **Precision Control:** Learning rate (5 decimals), Entropy (10 decimals)
- **Boundary Detection:** Tooltips automatically hide when outside chart area
- **Timestamp Conversion:** Seamless conversion between timesteps and timestamps
- **Error Handling:** Graceful handling of missing or invalid data
- **Configuration Display:** Shows current training configuration and parameters

### **Training Metrics**
- **Performance Tracking:** Rewards, loss, balance, equity, position
- **Trading Metrics:** Total trades, commission, PnL (realized/unrealized)
- **Learning Metrics:** Learning rate, entropy with high precision
- **Time Tracking:** Accurate training duration in seconds
- **Market Cycle Metrics:** Episode length tracking with quarterly cycles

### **Data Processing**
- **Missing Data Handling:** Interpolation and forward-fill methods
- **Technical Indicators:** Moving averages, RSI, Ichimoku cloud
- **Data Validation:** Duplicate removal and timestamp alignment
- **GPU Acceleration:** PyTorch with CUDA support for faster training

## 📈 API Endpoints

- `GET /` - RL training dashboard
- `GET /chart` - TradingView chart viewer
- `GET /api/training_metrics` - Raw training metrics JSON
- `GET /api/training_config` - Current training configuration
- `GET /api/candles` - OHLCV and indicator data

## 🎯 Usage Examples

### **Monitor Training Progress**
```bash
# Start training with quarterly episodes (auto-resumes from checkpoint)
python scripts/run_rl_pipeline.py

# Start training from scratch
python scripts/run_rl_pipeline.py --fresh-start

# CPU-only training
python scripts/run_rl_pipeline.py --cpu-only

# In another terminal, start dashboard
uvicorn src.web.main:app --reload

# Open browser to http://localhost:8000/
```

### **Manage Checkpoints**
```bash
# List all available checkpoints
python scripts/manage_checkpoints.py list

# Remove all checkpoints (start fresh)
python scripts/manage_checkpoints.py clear

# Backup all checkpoints
python scripts/manage_checkpoints.py backup

# Show latest checkpoint info
python scripts/manage_checkpoints.py latest
```

### **Experiment with Different Configurations**
```bash
# Test different market cycles
python -c "
from src.config.config import print_available_configs
print_available_configs()
"

# Use ultra-aggressive configuration
python -c "
from src.config.config import get_config
config = get_config('ultra_aggressive')
print(f'Ultra-aggressive config: {config}')
"
```

### **Calculate Optimal Timesteps**
```bash
# Calculate optimal training parameters
python scripts/calculate_timesteps.py
```

### **View Market Data**
```bash
# Start server
uvicorn src.web.main:app --reload

# Open browser to http://localhost:8000/chart
```

### **API Access**
```bash
# Get raw training metrics
curl http://localhost:8000/api/training_metrics

# Get current configuration
curl http://localhost:8000/api/training_config

# Get chart data
curl http://localhost:8000/api/candles
```

## 🔍 Troubleshooting

### **Dashboard Not Updating**
- Check if training is running and generating `outputs/plots/training_metrics.json`
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

### **GPU Issues**
- Use `--cpu-only` flag if GPU acceleration causes problems
- Check PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### **Training Issues**
- Use `--fresh-start` to clear all checkpoints and start over
- Enable `--debug` for detailed logging
- Check available configurations with `python -c "from src.config.config import print_available_configs; print_available_configs()"`

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
- **[PyTorch](https://pytorch.org/)** - Deep learning framework with GPU support

## 📄 License

MIT License - see LICENSE file for details.

---

**Ready to train your RL trading agent with market cycle optimization? Start with `python scripts/run_rl_pipeline.py` and monitor progress at `http://localhost:8000/`!** 🚀 