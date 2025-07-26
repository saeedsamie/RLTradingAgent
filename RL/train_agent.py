import torch
from stable_baselines3 import PPO

from RL.trading_env import TradingEnv


def train_agent(train_df, model_path='ppo_trading.zip', window_size=200, total_timesteps=5_000_000, debug=True,
                max_episode_steps=5_000_000):
    """Train PPO agent on the trading environment and save the model. Uses MlpPolicy."""
    # Select only the allowed feature columns and 'close' for price
    feature_cols = [
        'ma20', 'ma50', 'ma200', 'rsi', 'ichimoku_conversion', 'ichimoku_base',
        'ichimoku_leading_a', 'ichimoku_leading_b', 'ichimoku_chikou',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'atr', 'obv'
    ]
    # Keep 'close' for price calculation in env
    filtered_df = train_df[['close'] + feature_cols].copy()
    env = TradingEnv(filtered_df, window_size=window_size, debug=debug, max_episode_steps=max_episode_steps)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=1e-3,
        clip_range=0.2
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f'Model saved to {model_path}')
    return model
