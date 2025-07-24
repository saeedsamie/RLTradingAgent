from stable_baselines3 import PPO

from RL.trading_env import TradingEnv


def train_agent(train_df, model_path='ppo_trading.zip', window_size=50, total_timesteps=100_000):
    """Train PPO agent on the trading environment and save the model."""
    env = TradingEnv(train_df, window_size=window_size)
    model = PPO('CCNPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f'Model saved to {model_path}')
    return model
