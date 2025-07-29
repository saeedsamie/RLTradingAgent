import pandas as pd
import numpy as np
from RL.train_agent import train_agent

def generate_sample_data(n=1000):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n, freq='5min')
    data = {
        'datetime': dates,
        'open': np.random.uniform(1200, 1300, n),
        'high': np.random.uniform(1200, 1300, n),
        'low': np.random.uniform(1200, 1300, n),
        'close': np.random.uniform(1200, 1300, n),
        'volume': np.random.uniform(80, 120, n),
        'ma20': np.random.uniform(0.05, 0.06, n),
        'ma50': np.random.uniform(0.05, 0.06, n),
        'ma200': np.random.uniform(0.05, 0.06, n),
        'rsi': np.random.uniform(30, 70, n),
        'ichimoku_conversion': np.random.uniform(0.05, 0.06, n),
        'ichimoku_base': np.random.uniform(0.05, 0.06, n),
        'ichimoku_leading_a': np.random.uniform(0.05, 0.06, n),
        'ichimoku_leading_b': np.random.uniform(0.05, 0.06, n),
        'ichimoku_chikou': np.random.uniform(0.05, 0.06, n),
        'MACD_12_26_9': np.random.uniform(-0.01, 0.01, n),
        'MACDh_12_26_9': np.random.uniform(-0.01, 0.01, n),
        'MACDs_12_26_9': np.random.uniform(-0.01, 0.01, n),
        'STOCHk_14_3_3': np.random.uniform(20, 80, n),
        'STOCHd_14_3_3': np.random.uniform(20, 80, n),
        'atr': np.random.uniform(0.02, 0.04, n),
        'obv': np.random.uniform(0.1, 0.2, n)
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating sample data...")
    df = generate_sample_data(1000)
    print("Starting training and profiling...")
    # Use small timesteps for quick profiling
    train_agent(df, model_path='models/ppo_trading_profiled.zip', window_size=50, total_timesteps=2000, debug=False, max_episode_steps=200)
    print("Profiling complete.") 