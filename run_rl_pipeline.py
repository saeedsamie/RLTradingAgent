from RL.evaluate import evaluate_agent, sharpe_ratio, max_drawdown
from RL.plotting import plot_equity_curve
from RL.train_agent import train_agent
from data_prep import load_data, check_missing_intervals

DATA_PATH = 'dataset/xauusd_5m_alpari_filled_indicated.csv'
MODEL_PATH = 'ppo_trading.zip'
WINDOW_SIZE = 50
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
    model = train_agent(train_df, model_path=MODEL_PATH, window_size=WINDOW_SIZE)

    # Evaluate agent
    rewards, equity_curve = evaluate_agent(MODEL_PATH, test_df, window_size=WINDOW_SIZE)
    print(f'Cumulative return: {equity_curve[-1] - equity_curve[0]:.2f}')
    print(f'Sharpe ratio: {sharpe_ratio(rewards):.2f}')
    print(f'Max drawdown: {max_drawdown(equity_curve):.2%}')

    # Plot results
    plot_equity_curve(equity_curve)
