import numpy as np
import torch
from stable_baselines3 import PPO

from RL.trading_env import TradingEnv


def evaluate_agent(model_path, test_df, window_size=50):
    """Run the trained agent on test data and collect results. Uses MlpPolicy."""
    env = TradingEnv(test_df, window_size=window_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO.load(model_path, device=device)
    obs, _ = env.reset()
    done = False
    rewards = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
    equity_curve = env.equity_curve
    return rewards, equity_curve


def sharpe_ratio(returns, risk_free_rate=0):
    returns = np.array(returns)
    if returns.std() == 0:
        return 0
    return (returns.mean() - risk_free_rate) / (returns.std() + 1e-8) * np.sqrt(252 * 24 * 12)


def max_drawdown(equity_curve):
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown.min()
