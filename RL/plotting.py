import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_equity_curve(
        equity_curve,
        rewards=None,
        returns=None,
        drawdown=None,
        window=50
):
    # Ensure the 'plots' directory exists
    os.makedirs('plots', exist_ok=True)

    # Equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve)
    plt.title('Equity Curve')
    plt.xlabel('Step')
    plt.ylabel('Equity')
    plt.grid()
    plt.tight_layout()
    plt.savefig('../plots/equity_curve.png')
    plt.close()

    # Rewards over time
    if rewards is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(rewards)
        plt.title('Rewards Over Time')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.grid()
        plt.tight_layout()
        plt.savefig('../plots/rewards_over_time.png')
        plt.close()

    # Drawdown curve
    if drawdown is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(drawdown)
        plt.title('Drawdown Curve')
        plt.xlabel('Step')
        plt.ylabel('Drawdown')
        plt.grid()
        plt.tight_layout()
        plt.savefig('../plots/drawdown_curve.png')
        plt.close()

    # Histogram of returns
    if returns is not None:
        plt.figure(figsize=(8, 4))
        plt.hist(returns, bins=50, alpha=0.7)
        plt.title('Histogram of Returns')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig('../plots/returns_histogram.png')
        plt.close()

    # Rolling Sharpe ratio
    if returns is not None:
        rolling_mean = pd.Series(returns).rolling(window).mean()
        rolling_std = pd.Series(returns).rolling(window).std()
        rolling_sharpe = rolling_mean / (rolling_std + 1e-8) * np.sqrt(window)
        plt.figure(figsize=(10, 4))
        plt.plot(rolling_sharpe)
        plt.title(f'Rolling Sharpe Ratio (window={window})')
        plt.xlabel('Step')
        plt.ylabel('Sharpe Ratio')
        plt.grid()
        plt.tight_layout()
        plt.savefig('../plots/rolling_sharpe_ratio.png')
        plt.close()

    # Rolling volatility
    if returns is not None:
        rolling_vol = pd.Series(returns).rolling(window).std()
        plt.figure(figsize=(10, 4))
        plt.plot(rolling_vol)
        plt.title(f'Rolling Volatility (window={window})')
        plt.xlabel('Step')
        plt.ylabel('Volatility')
        plt.grid()
        plt.tight_layout()
        plt.savefig('../plots/rolling_volatility.png')
        plt.close()

    # Cumulative rewards
    if rewards is not None:
        cum_rewards = np.cumsum(rewards)
        plt.figure(figsize=(10, 4))
        plt.plot(cum_rewards)
        plt.title('Cumulative Rewards')
        plt.xlabel('Step')
        plt.ylabel('Cumulative Reward')
        plt.grid()
        plt.tight_layout()
        plt.savefig('../plots/cumulative_rewards.png')
        plt.close()

    # Scatter plot: returns vs. equity
    if returns is not None and equity_curve is not None:
        min_len = min(len(returns), len(equity_curve))
        plt.figure(figsize=(8, 6))
        plt.scatter(returns[:min_len], np.array(equity_curve)[:min_len], alpha=0.5)
        plt.title('Returns vs. Equity')
        plt.xlabel('Return')
        plt.ylabel('Equity')
        plt.tight_layout()
        plt.savefig('../plots/returns_vs_equity_scatter.png')
        plt.close()

    # Moving average of equity
    if equity_curve is not None:
        equity_ma = pd.Series(equity_curve).rolling(window).mean()
        plt.figure(figsize=(10, 5))
        plt.plot(equity_curve, label='Equity')
        plt.plot(equity_ma, label=f'MA{window}')
        plt.title(f'Equity Curve with {window}-Step Moving Average')
        plt.xlabel('Step')
        plt.ylabel('Equity')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('../plots/equity_curve_with_ma.png')
        plt.close()
