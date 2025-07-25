import torch
from stable_baselines3 import PPO
from RL.custom_mixed_policy import MixedActionPolicy
from RL.trading_env import TradingEnv


def train_agent(train_df, model_path='ppo_trading.zip', window_size=1000, total_timesteps=100_000):
    """Train PPO agent on the trading environment and save the model. Uses MixedActionPolicy."""
    env = TradingEnv(train_df, window_size=window_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PPO(MixedActionPolicy, env, verbose=1, device=device)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f'Model saved to {model_path}')
    return model
