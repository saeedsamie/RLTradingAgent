import numpy as np

from RL.evaluate import evaluate_agent, sharpe_ratio, max_drawdown
from RL.plotting import plot_equity_curve
from RL.train_agent import train_agent
from scripts.data_prep import load_data, check_missing_intervals
from scripts.config import get_config, print_available_configs

# Configuration - using quarterly market cycles for better pattern recognition
DATA_PATH = 'data/processed/xauusd_5m_alpari_normalized.csv'
MODEL_PATH = 'models/ppo_trading.zip'

config = get_config('quarterly_focused')
WINDOW_SIZE = config['window_size']
MAX_EPISODE_STEPS = config['max_episode_steps']
TOTAL_TIMESTEPS = config['total_timesteps']
TRAIN_RATIO = config['train_ratio']

# Add command line argument for fresh start
import sys
FRESH_START = '--fresh-start' in sys.argv

if __name__ == '__main__':
    print(f"Using configuration: {config['description']}")
    print(f"Window size: {WINDOW_SIZE} bars ({WINDOW_SIZE/288:.1f} days)")
    print(f"Episode length: {MAX_EPISODE_STEPS} bars ({MAX_EPISODE_STEPS/288:.1f} days)")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,} ({TOTAL_TIMESTEPS/1000000:.1f}M)")
    print()
    
    # Load data
    df = load_data(DATA_PATH)
    print(f'Dataset loaded: {df.shape[0]} rows')
    # Check for missing intervals
    check_missing_intervals(df, time_col=df.columns[0], freq='5min')

    # Split train/test
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    print(f'Train: {train_df.shape[0]}, Test: {test_df.shape[0]}')

    # Handle fresh start option
    if FRESH_START:
        import os
        import glob
        print("Fresh start requested. Clearing existing checkpoints...")
        checkpoint_files = glob.glob('models/checkpoints/ppo_trading_*_steps.zip')
        for file in checkpoint_files:
            os.remove(file)
            print(f"Removed: {file}")
        print("Starting training from scratch.")

    # Train agent
    model = train_agent(
        train_df,
        model_path=MODEL_PATH,
        window_size=WINDOW_SIZE,
        max_episode_steps=MAX_EPISODE_STEPS,
        total_timesteps=TOTAL_TIMESTEPS,
        debug=False
    )

    # Evaluate agent
    rewards, equity_curve = evaluate_agent(
        MODEL_PATH,
        test_df,
        window_size=WINDOW_SIZE,
        max_episode_steps=MAX_EPISODE_STEPS
    )
    print(f'Cumulative return: {equity_curve[-1] - equity_curve[0]:.2f}')
    print(f'Sharpe ratio: {sharpe_ratio(rewards):.2f}')
    print(f'Max drawdown: {max_drawdown(equity_curve):.2%}')

    # Prepare additional data for plotting
    returns = np.array(rewards)
    equity_curve_np = np.array(equity_curve)
    drawdown = (equity_curve_np - np.maximum.accumulate(equity_curve_np)) / np.maximum.accumulate(equity_curve_np)
    window = 50

    # Plot results (all advanced plots)
    plot_equity_curve(
        equity_curve,
        rewards=rewards,
        returns=returns,
        drawdown=drawdown,
        window=window
    )
