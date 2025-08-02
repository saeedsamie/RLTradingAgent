import json
import os
import time
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback

from RL.trading_env import TradingEnv


class TrainingMetricsCallback(BaseCallback):
    def __init__(self, log_path='plots/training_metrics.json', verbose=0, save_freq=100000, total_timesteps=None, resume_timesteps=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.save_freq = save_freq
        self.total_timesteps = total_timesteps
        self.resume_timesteps = resume_timesteps  # Timesteps already completed
        self.metrics = []
        self.recent_rewards = []
        self.recent_lengths = []
        self.recent_trades = []  # Track recent trading activity
        self.saved_metrics_num = 0
        self.window = 100  # For moving averages
        self.training_start_time = time.time()  # Record training start time
        self.conservative_warnings = 0  # Track conservative behavior warnings
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _on_step(self) -> bool:
        # Log episode reward, length, loss, custom info fields, learning rate, and entropy at the end of each episode
        episode_data_found = False
        
        # Try to get loss values from the model's training process
        current_loss = None
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            logger_values = self.model.logger.name_to_value
            # Try to get the most recent loss value
            current_loss = (
                logger_values.get('train/policy_loss', None) or
                logger_values.get('train/value_loss', None) or
                logger_values.get('train/loss', None)
            )
        
        # Also try to get loss from the model's internal state if available
        if current_loss is None and hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
            try:
                # This is a more direct approach to get loss values
                if hasattr(self.model, '_last_loss'):
                    current_loss = self.model._last_loss
            except Exception:
                pass
        
        # Try to get loss from the model's training logs
        if current_loss is None and hasattr(self.model, 'logger'):
            try:
                # Access the logger's recorded values
                if hasattr(self.model.logger, 'record') and hasattr(self.model.logger, 'name_to_value'):
                    # Look for any loss-related keys
                    for key, value in self.model.logger.name_to_value.items():
                        if 'loss' in key.lower() and value is not None:
                            current_loss = value
                            break
            except Exception:
                pass
        
        # Try to get loss from the model's training process by accessing the model's internal state
        if current_loss is None:
            try:
                # In PPO, loss values are typically available through the policy's training process
                if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                    # Try to access the last computed loss
                    if hasattr(self.model, 'train') and hasattr(self.model.train, 'loss'):
                        current_loss = self.model.train.loss
            except Exception:
                pass
        
        # Debug: Print all available logger keys to understand what's available
        if self.verbose > 0 and self.num_timesteps % 10000 == 0:
            if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                print(f"Available logger keys at timestep {self.num_timesteps}:")
                for key, value in self.model.logger.name_to_value.items():
                    print(f"  {key}: {value}")
        
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
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

                    episode_data_found = True
                    if self.verbose > 0:
                        print(f"Episode data found at timestep {self.num_timesteps}")
                    reward = info['episode']['r']
                    length = info['episode']['l']
                    self.recent_rewards.append(reward)
                    self.recent_lengths.append(length)
                    self.recent_trades.append(total_trades)
                    if len(self.recent_rewards) > self.window:
                        self.recent_rewards.pop(0)
                    if len(self.recent_lengths) > self.window:
                        self.recent_lengths.pop(0)
                    if len(self.recent_trades) > self.window:
                        self.recent_trades.pop(0)
                    mean_reward = sum(self.recent_rewards) / len(self.recent_rewards)
                    mean_length = sum(self.recent_lengths) / len(self.recent_lengths)
                    mean_trades = sum(self.recent_trades) / len(self.recent_trades) if self.recent_trades else 0

                    # Learning rate
                    lr = None
                    if hasattr(self.model, 'lr_schedule'):
                        try:
                            lr = float(self.model.lr_schedule(self.num_timesteps))
                        except Exception:
                            pass
                    # Entropy and Loss (if available from logger)
                    entropy = None
                    loss_from_logger = None
                    if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
                        logger_values = self.model.logger.name_to_value
                        entropy = logger_values.get('train/entropy_loss', None)
                        # Try different loss keys that might be available
                        loss_from_logger = (
                            logger_values.get('train/policy_loss', None) or
                            logger_values.get('train/value_loss', None) or
                            logger_values.get('train/loss', None)
                        )
                    # Use loss from logger if available, otherwise use from info
                    final_loss = loss_from_logger if loss_from_logger is not None else loss
                    
                    # If we still don't have loss, try the current_loss from the beginning of the step
                    if final_loss is None:
                        final_loss = current_loss
                    
                    # Debug logging for loss values
                    if self.verbose > 0 and (loss_from_logger is not None or loss is not None or current_loss is not None):
                        print(f"Loss values at timestep {self.num_timesteps}:")
                        print(f"  - From logger: {loss_from_logger}")
                        print(f"  - From info: {loss}")
                        print(f"  - Current loss: {current_loss}")
                        print(f"  - Final loss: {final_loss}")
                    # Calculate elapsed training time
                    elapsed_time = time.time() - self.training_start_time
                    
                    # Calculate progress percentage including resumed timesteps
                    total_completed_timesteps = self.num_timesteps + self.resume_timesteps
                    progress_percentage = (total_completed_timesteps / self.total_timesteps * 100) if hasattr(self, 'total_timesteps') else 0
                    
                    # Check for conservative behavior with more aggressive warnings
                    if mean_trades < 1.0 and len(self.recent_trades) >= 5:
                        self.conservative_warnings += 1
                        if self.conservative_warnings <= 5:  # More warnings
                            print(f"🚨 CRITICAL: Agent not trading! Average trades: {mean_trades:.1f}")
                            print(f"   This indicates the reward function needs to be more aggressive")
                            print(f"   Consider restarting with even stronger trading incentives")
                    
                    self.metrics.append({
                        'timesteps': self.num_timesteps + self.resume_timesteps,  # Include resumed timesteps
                        'reward': reward,
                        'length': length,
                        'mean_reward_100': mean_reward,
                        'mean_length_100': mean_length,
                        'mean_trades_100': mean_trades,  # Add mean trades tracking
                        'loss': final_loss,
                        'balance': balance,
                        'equity': equity,
                        'position': position,
                        'total_trades': total_trades,
                        'total_commission': total_commission,
                        'total_pnl': total_pnl,
                        'unrealized_pnl': unrealized_pnl,
                        'commission': commission,
                        'learning_rate': lr,
                        'entropy': entropy,
                        'training_time_seconds': elapsed_time,
                        'progress_percentage': progress_percentage
                    })
                    self._save_metrics()

        # Save metrics periodically during training based on timesteps, not just when episode data is available
        # if self.saved_metrics_num < len(self.metrics):
        #     self.saved_metrics_num = len(self.metrics)
        #     self._save_metrics()

        return True

    def _save_metrics(self):
        """Save current metrics to file"""
        print(f"Saving metrics at timestep {self.num_timesteps} with {len(self.metrics)} data points",
              self.saved_metrics_num)
        with open(self.log_path, 'w') as f:
            json.dump(self.metrics, f)

    def _on_training_end(self) -> None:
        # Save metrics to file at the end of training
        self._save_metrics()


def train_agent(train_df, model_path='ppo_trading.zip', window_size=288, total_timesteps=5_000_000, debug=False,
                max_episode_steps=25920, checkpoint_dir='models/checkpoints', checkpoint_freq=100000):
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
    
    
    # Ensure datetime index is preserved - check both index and first column
    # Assume the first column is 'datetime' and set it as the index

    filtered_df = train_df[['close'] + feature_cols].copy()
    filtered_df.index = pd.to_datetime(train_df['datetime'])
    
    env = TradingEnv(filtered_df, window_size=window_size, debug=debug, max_episode_steps=max_episode_steps)
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        # Test CUDA functionality
        test_tensor = torch.randn(100, 100).cuda()
        print(f"CUDA test tensor device: {test_tensor.device}")
        print(f"CUDA memory after test: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    
    # Check for existing checkpoints to resume training
    import glob
    import re
    
    # Find the latest checkpoint
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'ppo_trading_*_steps.zip'))
    latest_checkpoint = None
    latest_timesteps = 0
    
    if checkpoint_files:
        for checkpoint_file in checkpoint_files:
            # Extract timesteps from filename (e.g., ppo_trading_100000_steps.zip -> 100000)
            match = re.search(r'ppo_trading_(\d+)_steps\.zip', checkpoint_file)
            if match:
                timesteps = int(match.group(1))
                if timesteps > latest_timesteps:
                    latest_timesteps = timesteps
                    latest_checkpoint = checkpoint_file
    
    # ULTRA AGGRESSIVE PPO configuration to force trading
    print(f"Creating PPO model with device: {device}")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=1e-3,  # Balanced learning rate for deep network
        clip_range=0.2,  # Stable clip range for deep network
        ent_coef=0.3,  # Balanced entropy coefficient for exploration
        vf_coef=0.5,  # Higher value function coefficient for deep network
        max_grad_norm=0.5,  # Lower gradient clipping for stable deep learning
        n_steps=2048,  # Larger batch for deep network stability
        batch_size=64,  # Larger batch size for deep network
        n_epochs=4,  # More epochs for deep network learning
        gamma=0.99,  # Higher discount factor for deep network
        gae_lambda=0.95,  # Higher GAE lambda for deep network
        target_kl=0.01,  # Lower KL divergence target for stable deep learning
        tensorboard_log="./logs/",  # Enable tensorboard logging
        policy_kwargs={
            "net_arch": dict(pi=[512, 512, 256, 256, 128], vf=[512, 512, 256, 256, 128]),  # Deep network for complex patterns
            "activation_fn": torch.nn.Tanh,  # Tanh for better gradient flow
            "ortho_init": True,  # Orthogonal initialization for better training
        }
    )
    
    # Load latest checkpoint if available
    if latest_checkpoint and os.path.exists(latest_checkpoint):
        print(f"Found checkpoint: {latest_checkpoint}")
        print(f"Resuming training from {latest_timesteps:,} timesteps")
        model = PPO.load(latest_checkpoint, env=env, device=device)
        remaining_timesteps = total_timesteps - latest_timesteps
        print(f"Remaining timesteps to train: {remaining_timesteps:,}")
    else:
        print("No checkpoint found. Starting training from scratch.")
        remaining_timesteps = total_timesteps
    
    # Verify model is on the correct device
    print(f"Model device: {next(model.policy.parameters()).device}")
    print(f"Policy device: {model.policy.device}")
    print(f"Value function device: {next(model.policy.value_net.parameters()).device}")
    if torch.cuda.is_available():
        print(f"CUDA memory after model creation: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    callback = TrainingMetricsCallback(log_path='plots/training_metrics.json', save_freq=100, verbose=1, total_timesteps=total_timesteps, resume_timesteps=latest_timesteps)
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_dir,
                                             name_prefix='ppo_trading')
    start_train = time.time()
    model.learn(total_timesteps=remaining_timesteps, callback=[callback, checkpoint_callback])
    elapsed_train = time.time() - start_train
    model.save(model_path)
    print(f'Model saved to {model_path}')
    print(f'Total model.learn() time: {elapsed_train:.2f} seconds')
    if hasattr(env, 'step_times') and len(env.step_times) > 0:
        print(f'Average env.step() time: {sum(env.step_times) / len(env.step_times):.6f} seconds')
    if hasattr(env, 'reset_times') and len(env.reset_times) > 0:
        print(f'Average env.reset() time: {sum(env.reset_times) / len(env.reset_times):.6f} seconds')
    return model
