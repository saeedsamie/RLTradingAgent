import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import json
import os
import time
from stable_baselines3.common.callbacks import CheckpointCallback

from RL.trading_env import TradingEnv


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, log_path='plots/training_metrics.json', verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.metrics = []
        self.recent_rewards = []
        self.recent_lengths = []
        self.window = 100  # For moving averages
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _on_step(self) -> bool:
        # Log episode reward, length, loss, custom info fields, learning rate, and entropy at the end of each episode
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    reward = info['episode']['r']
                    length = info['episode']['l']
                    self.recent_rewards.append(reward)
                    self.recent_lengths.append(length)
                    if len(self.recent_rewards) > self.window:
                        self.recent_rewards.pop(0)
                    if len(self.recent_lengths) > self.window:
                        self.recent_lengths.pop(0)
                    mean_reward = sum(self.recent_rewards) / len(self.recent_rewards)
                    mean_length = sum(self.recent_lengths) / len(self.recent_lengths)
                    # Try to get loss if available (not always present)
                    loss = info.get('loss', None)
                    # Custom info fields from env
                    balance = info.get('balance', None)
                    equity = info.get('equity', None)
                    position = info.get('position', None)
                    total_trades = info.get('total_trades', None)
                    total_commission = info.get('total_commission', None)
                    total_pnl = info.get('total_pnl', None)
                    unrealized_pnl = info.get('unrealized_pnl', None)
                    price = info.get('price', None)
                    commission = info.get('commission', None)
                    # Learning rate
                    lr = None
                    if hasattr(self.model, 'lr_schedule'):
                        try:
                            lr = float(self.model.lr_schedule(self.num_timesteps))
                        except Exception:
                            pass
                    # Entropy (if available from logger)
                    entropy = None
                    if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                        entropy = self.model.logger.name_to_value.get('train/entropy_loss', None)
                    self.metrics.append({
                        'timesteps': self.num_timesteps,
                        'reward': reward,
                        'length': length,
                        'mean_reward_100': mean_reward,
                        'mean_length_100': mean_length,
                        'loss': loss,
                        'balance': balance,
                        'equity': equity,
                        'position': position,
                        'total_trades': total_trades,
                        'total_commission': total_commission,
                        'total_pnl': total_pnl,
                        'unrealized_pnl': unrealized_pnl,
                        'price': price,
                        'commission': commission,
                        'learning_rate': lr,
                        'entropy': entropy
                    })
        return True

    def _on_training_end(self) -> None:
        # Save metrics to file at the end of training
        with open(self.log_path, 'w') as f:
            json.dump(self.metrics, f)


def train_agent(train_df, model_path='ppo_trading.zip', window_size=200, total_timesteps=5_000_000, debug=True,
                max_episode_steps=10000, checkpoint_dir='models/checkpoints', checkpoint_freq=100000):
    """Train PPO agent on the trading environment and save the model. Uses MlpPolicy.
    Args:
        train_df: DataFrame with training data.
        model_path: Path to save the final model.
        window_size: Observation window size.
        total_timesteps: Total timesteps to train.
        debug: Enable debug logging.
        max_episode_steps: Max steps per episode.
        checkpoint_dir: Directory to save checkpoints.
        checkpoint_freq: Save checkpoint every N steps.
    """
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
        learning_rate=1e-4,
        clip_range=0.2
    )
    callback = TrainingMetricsCallback(log_path='plots/training_metrics.json')
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_dir, name_prefix='ppo_trading')
    start_train = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[callback, checkpoint_callback])
    elapsed_train = time.time() - start_train
    model.save(model_path)
    print(f'Model saved to {model_path}')
    print(f'Total model.learn() time: {elapsed_train:.2f} seconds')
    if hasattr(env, 'step_times') and len(env.step_times) > 0:
        print(f'Average env.step() time: {sum(env.step_times)/len(env.step_times):.6f} seconds')
    if hasattr(env, 'reset_times') and len(env.reset_times) > 0:
        print(f'Average env.reset() time: {sum(env.reset_times)/len(env.reset_times):.6f} seconds')
    return model
