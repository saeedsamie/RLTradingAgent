import numpy as np

from RL.evaluate import evaluate_agent, sharpe_ratio, max_drawdown
from RL.plotting import plot_equity_curve
from RL.train_agent import train_agent
from scripts.data_prep import load_data, check_missing_intervals

DATA_PATH = 'data/processed/xauusd_5m_alpari_filled_indicated.csv'
MODEL_PATH = 'models/ppo_trading.zip'
WINDOW_SIZE = 200
MAX_EPISODE_STEPS = 10000
TRAIN_RATIO = 0.8

if __name__ == '__main__':
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

    # Train agent
    model = train_agent(
        train_df,
        model_path=MODEL_PATH,
        window_size=WINDOW_SIZE,
        max_episode_steps=MAX_EPISODE_STEPS
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
