import numpy as np
import torch
from stable_baselines3 import PPO

from RL.custom_mixed_policy import MixedActionPolicy
from RL.trading_env import TradingEnv


def evaluate_agent(model_path, test_df, window_size=50, max_episode_steps=10000):
    """Run the trained agent on test data and collect results. Uses MixedActionPolicy."""
    # Select only the allowed feature columns and 'close' for price
    feature_cols = [
        'ma20', 'ma50', 'ma200', 'rsi', 'ichimoku_conversion', 'ichimoku_base',
        'ichimoku_leading_a', 'ichimoku_leading_b', 'ichimoku_chikou',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'atr', 'obv'
    ]
    # Keep 'close' for price calculation in env
    filtered_df = test_df[['close'] + feature_cols].copy()
    env = TradingEnv(filtered_df, window_size=window_size, debug=False, max_episode_steps=max_episode_steps)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO.load(model_path, device=device, custom_policy=MixedActionPolicy)
    obs, _ = env.reset()
    done = False
    rewards = []
    while not done:
        # Model outputs a tensor of shape (2,) [action, confidence]
        action_tensor, _ = model.predict(obs, deterministic=True)
        # Convert to tuple for env.step
        if isinstance(action_tensor, np.ndarray):
            act = int(action_tensor[0])
            conf = float(action_tensor[1])
            action = (act, conf)
        else:
            action = (int(action_tensor[0]), float(action_tensor[1]))
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
