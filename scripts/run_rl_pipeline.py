import json
import os
import time
import csv

import numpy as np
import pandas as pd
from RL.trading_env_improved import ImprovedTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch

from RL.evaluate import evaluate_with_trade_analysis
from RL.plotting import plot_equity_curve
from scripts.config import get_config
from scripts.data_prep import load_data, check_missing_intervals

# Configuration - using quarterly market cycles for better pattern recognition
DATA_PATH = 'data/processed/xauusd_5m_alpari_normalized_ticksize.csv'
MODEL_PATH = 'models/ppo_trading.zip'

config = get_config('deep_network')
WINDOW_SIZE = config['window_size']
MAX_EPISODE_STEPS = config['max_episode_steps']
TOTAL_TIMESTEPS = config['total_timesteps']
TRAIN_RATIO = config['train_ratio']

# Add command line argument for fresh start
import sys
def handle_fresh_start():
            import os
            import glob
            import shutil

            print("Fresh start requested. Clearing existing checkpoints...")
            checkpoint_files = glob.glob('models/checkpoints/ppo_trading_*_steps.zip')
            for file in checkpoint_files:
                os.remove(file)
                print(f"Removed: {file}")
            print("Starting training from scratch.")

            # Clean up the 'plots' and 'trade_history' directories for a fresh start
            for dir_path in ['plots', 'trade_history']:
                if os.path.exists(dir_path):
                    print(f"Clearing directory: {dir_path}")
                    shutil.rmtree(dir_path)
                    print(f"Removed: {dir_path}")
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created empty directory: {dir_path}")


FRESH_START = '--fresh-start' in sys.argv
DEBUG_MODE = '--debug' in sys.argv

class TrainingMetricsCallback(BaseCallback):
    def __init__(self, log_path='plots/training_metrics.json', verbose=0, save_freq=1000, total_timesteps=None, resume_timesteps=0):
        """Initialize callback.
        
        Args:
            log_path: Path to save metrics JSON
            verbose: Verbosity level
            save_freq: Save metrics every N timesteps
            total_timesteps: Total timesteps for training
            resume_timesteps: Timesteps already completed if resuming training
        """
        super().__init__(verbose)
        print(f"\nInitializing TrainingMetricsCallback:")
        print(f"  Log path: {log_path}")
        print(f"  Save frequency: Every {save_freq:,} timesteps")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"  Resume timesteps: {resume_timesteps:,}")
        
        self.log_path = log_path
        self.save_freq = save_freq
        self.total_timesteps = total_timesteps
        self.resume_timesteps = resume_timesteps
        self.metrics = []
        self.recent_rewards = []
        self.recent_lengths = []
        self.recent_trades = []
        self.saved_metrics_num = 0
        self.window = 100
        self.training_start_time = time.time()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"  Created metrics directory: {os.path.dirname(log_path)}")
        
        # Save initial empty metrics file to ensure we can write to it
        self._save_metrics()
        print("  Created initial empty metrics file")

    def _on_step(self) -> bool:
        episode_data_found = False
        
        # Debug: Print step count every 1000 steps
        if self.num_timesteps % 1000 == 0:
            print(f"Callback step {self.num_timesteps:,}")
        
        # Try to get loss values from the model's training process
        current_loss = None
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            logger_values = self.model.logger.name_to_value
            current_loss = (
                logger_values.get('train/policy_loss', None) or
                logger_values.get('train/value_loss', None) or
                logger_values.get('train/loss', None)
            )
        
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    loss = info.get('loss', None)
                    balance = info.get('balance', None)
                    equity = info.get('equity', None)
                    position = info.get('position', None)
                    total_trades = info.get('total_trades', None)
                    total_commission = info.get('total_commission', None)
                    total_pnl = info.get('total_pnl', None)
                    unrealized_pnl = info.get('unrealized_pnl', None)
                    price = info.get('price', None)
                    commission = info.get('commission', None)
                    trades_this_episode = info.get('trades_this_episode', None)
                    max_drawdown = info.get('max_drawdown', None)
                    consecutive_wins = info.get('consecutive_wins', None)
                    consecutive_losses = info.get('consecutive_losses', None)
                    position_duration = info.get('position_duration', None)
                    last_trade_pnl = info.get('last_trade_pnl', None)

                    episode_data_found = True
                    print(f"Episode data found at timestep {self.num_timesteps:,}")
                    print(f"  Reward: {info['episode']['r']:.2f}")
                    print(f"  Length: {info['episode']['l']}")
                    print(f"  Total trades: {total_trades}")
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

                    # Calculate moving averages
                    mean_reward = np.mean(self.recent_rewards) if self.recent_rewards else reward
                    mean_length = np.mean(self.recent_lengths) if self.recent_lengths else length
                    mean_trades = np.mean(self.recent_trades) if self.recent_trades else total_trades

                    # Get learning rate and entropy
                    learning_rate = self.model.learning_rate
                    entropy = info.get('entropy', None)

                    # Calculate progress percentage
                    progress_percentage = (self.num_timesteps + self.resume_timesteps) / self.total_timesteps if self.total_timesteps else 0

                    # Create metric entry
                    metric_entry = {
                        'timesteps': self.num_timesteps + self.resume_timesteps,
                        'reward': reward,
                        'length': length,
                        'mean_reward_100': mean_reward,
                        'mean_length_100': mean_length,
                        'mean_trades_100': mean_trades,
                        'loss': current_loss if current_loss is not None else 0.0,
                        'balance': balance,
                        'equity': equity,
                        'position': position,
                        'total_trades': total_trades,
                        'total_commission': total_commission,
                        'total_pnl': total_pnl,
                        'unrealized_pnl': unrealized_pnl,
                        'commission': commission,
                        'learning_rate': learning_rate,
                        'entropy': entropy,
                        'training_time_seconds': time.time() - self.training_start_time,
                        'progress_percentage': progress_percentage,
                        'trades_this_episode': trades_this_episode,
                        'max_drawdown': max_drawdown,
                        'consecutive_wins': consecutive_wins,
                        'consecutive_losses': consecutive_losses,
                        'position_duration': position_duration,
                        'last_trade_pnl': last_trade_pnl
                    }

                    self.metrics.append(metric_entry)

                    # Save metrics after each episode
                    print(f"\nSaving metrics at timestep {self.num_timesteps:,}:")
                    print(f"  Current metrics count: {len(self.metrics)}")
                    print(f"  Save path: {self.log_path}")
                    self._save_metrics()

                    # Save trade history for this episode using helper function
                    env = self.training_env.envs[0]
                    if hasattr(env, 'trade_history') and env.trade_history:
                        self._save_trade_history_for_episode(env, self.num_timesteps)

        return True

    def _save_trade_history_for_episode(self, env, timesteps):
        """Save trade history for this episode to a CSV file named with the current timestep"""
        if not env.trade_history:
            print("  No trades in this episode")
            return

        trades_csv_path = "trade_history/trades.csv"
        os.makedirs("trade_history", exist_ok=True)
                        
        # Append mode if file exists, write mode if it doesn't
        mode = "a" if os.path.exists(trades_csv_path) else "w"
        
        with open(trades_csv_path, mode=mode, newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header if it's a new file
            if mode == "w":
                writer.writerow([
                    "entry_idx", "exit_idx", "entry_datetime", "exit_datetime",
                    "entry_price", "exit_price", "entry_equity", "exit_equity",
                    "position_type", "pnl", "commission", "net_pnl",
                    "duration", "return_pct", "win", "reward", "episode"
                ])

            # Write trades for this episode
            trades_written = 0
            for trade in env.trade_history:
                # Skip trades that don't have both entry and exit times
                if not trade.get('entry_datetime') or not trade.get('exit_datetime'):
                    continue

                entry_datetime = trade['entry_datetime']
                exit_datetime = trade['exit_datetime']

                if hasattr(entry_datetime, 'strftime'):
                    entry_datetime = entry_datetime.strftime('%Y-%m-%d %H:%M:%S')
                if hasattr(exit_datetime, 'strftime'):
                    exit_datetime = exit_datetime.strftime('%Y-%m-%d %H:%M:%S')

                writer.writerow([
                    trade['entry_idx'], trade['exit_idx'],
                    entry_datetime, exit_datetime,
                    trade['entry_price'], trade['exit_price'],
                    trade['entry_equity'], trade['exit_equity'],
                    trade['position_type'], trade['pnl'],
                    trade['commission'], trade['net_pnl'],
                    trade['duration'], trade['return_pct'], trade['win'],
                    trade.get('reward', 0), timesteps
                ])
                trades_written += 1

        print(f"  Trade history appended to {trades_csv_path} ({trades_written} trades)")
        
        # Clear the trade history to prevent duplicates in next episode
        env.trade_history = []
        
    def _save_metrics(self):
        """Save metrics to JSON file."""
        try:
            print(f"Saving {len(self.metrics)} metrics to {self.log_path}")
            with open(self.log_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            self.saved_metrics_num += 1
            print(f"Metrics saved successfully to {self.log_path}")
        except Exception as e:
            print(f"Error saving metrics: {e}")
            import traceback
            traceback.print_exc()

    def _on_training_end(self) -> None:
        """Save metrics to file at the end of training."""
        self._save_metrics()





def max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve."""
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown.min()

def analyze_trades_improved(trades):
    """Analyze trades with improved metrics."""
    if not trades:
        return {
            'num_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'expectancy': 0,
            'total_commission': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_win_rate': 0,
            'short_win_rate': 0
        }
    
    num_trades = len(trades)
    winning_trades = [t for t in trades if t['win']]
    losing_trades = [t for t in trades if not t['win']]
    
    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
    avg_win = np.mean([t['net_pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['net_pnl'] for t in losing_trades]) if losing_trades else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss) if num_trades > 0 else 0
    
    total_commission = sum(t['commission'] for t in trades)
    
    long_trades = [t for t in trades if t['position_type'] == 'long']
    short_trades = [t for t in trades if t['position_type'] == 'short']
    
    long_win_rate = len([t for t in long_trades if t['win']]) / len(long_trades) if long_trades else 0
    short_win_rate = len([t for t in short_trades if t['win']]) / len(short_trades) if short_trades else 0
    
    return {
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
        'total_commission': total_commission,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'long_win_rate': long_win_rate,
        'short_win_rate': short_win_rate
    }

if __name__ == '__main__':
    # Ensure directories exist
    for dir_path in ['plots', 'trade_history', 'models/checkpoints']:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")
    
    print(f"\nUsing configuration: {config['description']}")
    print(f"Window size: {WINDOW_SIZE} bars ({WINDOW_SIZE / 288:.1f} days)")
    print(f"Episode length: {MAX_EPISODE_STEPS} bars ({MAX_EPISODE_STEPS / 288:.1f} days)")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,} ({TOTAL_TIMESTEPS / 1000000:.1f}M)")
    print()

    # Load data
    df = load_data(DATA_PATH)
    print(f'Dataset loaded: {df.shape[0]} rows')
    # Check for missing intervals
    check_missing_intervals(df, time_col=df.columns[0], freq='5min')

    # Split train/test while preserving datetime information
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f'Train: {train_df.shape[0]}, Test: {test_df.shape[0]}')

    # Print data ranges to verify no overlap
    if len(train_df) > 0 and len(test_df) > 0:
        train_start = train_df.iloc[0][train_df.columns[0]]
        train_end = train_df.iloc[-1][train_df.columns[0]]
        test_start = test_df.iloc[0][test_df.columns[0]]
        test_end = test_df.iloc[-1][test_df.columns[0]]
        print(f'Train period: {train_start} to {train_end}')
        print(f'Test period: {test_start} to {test_end}')
        print(f'Gap between train and test: {test_start - train_end}')
        if test_start <= train_end:
            print("WARNING: Test data overlaps with training data!")
        else:
            print("✓ Train/test split is correct - no overlap")

    # Handle fresh start option
    if FRESH_START:
        handle_fresh_start()

    # Train agent with improved environment
    print("\n=== Starting Training ===")
    print(f"Model path: {MODEL_PATH}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Max episode steps: {MAX_EPISODE_STEPS}")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"Debug mode: {DEBUG_MODE}")
    
    # Select only the allowed feature columns and 'close' for price
    feature_cols = [
        'ma20', 'ma50', 'ma200', 'rsi', 'ichimoku_conversion', 'ichimoku_base',
        'ichimoku_leading_a', 'ichimoku_leading_b', 'ichimoku_chikou',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'atr', 'obv'
    ]
    
    # Ensure datetime index is preserved
    filtered_df = train_df[['close'] + feature_cols].copy()
    filtered_df.index = pd.to_datetime(train_df['datetime'])
    
    # Create improved trading environment
    env = ImprovedTradingEnv(
        filtered_df, 
        window_size=WINDOW_SIZE, 
        debug=DEBUG_MODE, 
        max_episode_steps=MAX_EPISODE_STEPS,
        trading_frequency_penalty=0.1,
        risk_reward_ratio=2.0,
        profit_taking_multiplier=1.5,
        stop_loss_multiplier=1.0,
        holding_bonus=0.01,
        drawdown_penalty=0.5
    )
    
    print("\nEnvironment configuration:")
    print(f"Trading frequency penalty: {env.trading_frequency_penalty}")
    print(f"Risk-reward ratio: {env.risk_reward_ratio}")
    print(f"Profit taking multiplier: {env.profit_taking_multiplier}")
    print(f"Drawdown penalty: {env.drawdown_penalty}")
    
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=1e-3,
        clip_range=0.2,
        ent_coef=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        target_kl=0.01,
        tensorboard_log="./logs/",
        policy_kwargs={
            "net_arch": dict(pi=[512, 512, 256, 256, 128], vf=[512, 512, 256, 256, 128]),
            "activation_fn": torch.nn.Tanh,
            "ortho_init": True,
        }
    )
    
    # Create metrics callback with more frequent saving
    metrics_callback = TrainingMetricsCallback(
        log_path='plots/training_metrics.json',
        verbose=1,
        save_freq=1000,  # Save every 1000 timesteps
        total_timesteps=TOTAL_TIMESTEPS,
        resume_timesteps=0
    )
    print("\nMetrics callback configuration:")
    print(f"  Save frequency: Every 1,000 timesteps")
    print(f"  Log path: plots/training_metrics.json")
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='models/checkpoints',
        name_prefix="ppo_trading"
    )
    
    # Train the model
    print("\nStarting training...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[metrics_callback, checkpoint_callback],
            progress_bar=True
        )
    except ImportError as e:
        if "tqdm" in str(e) or "rich" in str(e):
            print("Progress bar not available, continuing without it...")
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=[metrics_callback, checkpoint_callback],
                progress_bar=False
            )
        else:
            raise e
    
    # Save the final model
    model.save(MODEL_PATH)
    print(f"\nTraining completed. Model saved to {MODEL_PATH}")

    # Evaluate agent with comprehensive trade analysis using improved environment
    print(f"\n=== Evaluation Debug Info ===")
    print(f"Model path: {MODEL_PATH}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Test data columns: {list(test_df.columns)}")
    print(f"Test data date range: {test_df.iloc[0][test_df.columns[0]]} to {test_df.iloc[-1][test_df.columns[0]]}")

    print("\n=== Starting Evaluation ===")
    print(f"Model path: {MODEL_PATH}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Max episode steps: {MAX_EPISODE_STEPS}")
    
    # Create evaluation environment
    filtered_test_df = test_df[['close'] + feature_cols].copy()
    filtered_test_df.index = pd.to_datetime(test_df['datetime'])
    
    env = ImprovedTradingEnv(
        filtered_test_df, 
        window_size=WINDOW_SIZE, 
        debug=DEBUG_MODE, 
        max_episode_steps=MAX_EPISODE_STEPS,
        trading_frequency_penalty=0.1,
        risk_reward_ratio=2.0,
        profit_taking_multiplier=1.5,
        stop_loss_multiplier=1.0,
        holding_bonus=0.01,
        drawdown_penalty=0.5
    )
    
    # Load model
    model = PPO.load(MODEL_PATH, env=env)
    
    # Run evaluation
    obs, _ = env.reset()
    done = False
    rewards = []
    
    print("\nRunning evaluation...")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        
        if info.get('total_trades', 0) % 100 == 0:
            print(f"Trades: {info.get('total_trades', 0)}, Balance: ${info.get('balance', 0):.2f}")
    
    # Get results
    results = {
        'rewards': rewards,
        'equity_curve': env.equity_curve,
        'trades': env.trade_history,
        'trade_stats': analyze_trades_improved(env.trade_history),
        'total_return': (env.balance - env.initial_balance) / env.initial_balance,
        'sharpe_ratio': np.mean(rewards) / (np.std(rewards) + 1e-8) * np.sqrt(252 * 24 * 12),
        'max_drawdown': env.max_drawdown
    }

    # Extract results
    rewards = results['rewards']
    equity_curve = results['equity_curve']
    trades = results['trades']
    trade_stats = results['trade_stats']
    
    # Save trade history to CSV
    if trades:
        trades_csv_path = "trade_history/trades.csv"
        os.makedirs("trade_history", exist_ok=True)
        
        with open(trades_csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "entry_idx", "exit_idx", "entry_datetime", "exit_datetime",
                "entry_price", "exit_price", "entry_equity", "exit_equity",
                "position_type", "pnl", "commission", "net_pnl",
                "duration", "return_pct", "win", "reward"
            ])

            for trade in trades:
                # Get datetime values from trade dictionary
                entry_datetime = trade.get('entry_datetime', "N/A")
                exit_datetime = trade.get('exit_datetime', "N/A")

                # Convert to string if it's a datetime object
                if hasattr(entry_datetime, 'strftime'):
                    entry_datetime = entry_datetime.strftime('%Y-%m-%d %H:%M:%S')
                if hasattr(exit_datetime, 'strftime'):
                    exit_datetime = exit_datetime.strftime('%Y-%m-%d %H:%M:%S')

                writer.writerow([
                    trade['entry_idx'], trade['exit_idx'],
                    entry_datetime, exit_datetime,
                    trade['entry_price'], trade['exit_price'],
                    trade['entry_equity'], trade['exit_equity'],
                    trade['position_type'], trade['pnl'],
                    trade['commission'], trade['net_pnl'],
                    trade['duration'], trade['return_pct'], trade['win'],
                    trade.get('reward', 0)
                ])
        print(f"Trade history saved to {trades_csv_path}")
    else:
        print("No trades found in evaluation results")

    # Print comprehensive results
    print(f'Cumulative return: {results["total_return"]:.2%}')
    print(f'Sharpe ratio: {results["sharpe_ratio"]:.2f}')
    print(f'Max drawdown: {results["max_drawdown"]:.2%}')
    print(f'Total trades: {trade_stats["num_trades"]}')
    print(f'Win rate: {trade_stats["win_rate"]:.2%}')
    print(f'Average win: ${trade_stats["avg_win"]:.2f}')
    print(f'Average loss: ${trade_stats["avg_loss"]:.2f}')
    print(f'Expectancy: ${trade_stats["expectancy"]:.2f}')
    print(f'Total commission: ${trade_stats["total_commission"]:.2f}')
    print(f'Long trades: {trade_stats["long_trades"]}, Win rate: {trade_stats["long_win_rate"]:.2%}')
    print(f'Short trades: {trade_stats["short_trades"]}, Win rate: {trade_stats["short_win_rate"]:.2%}')

    # Prepare additional data for plotting
    returns = np.array(rewards)
    equity_curve_np = np.array(equity_curve)
    drawdown = (equity_curve_np - np.maximum.accumulate(equity_curve_np)) / np.maximum.accumulate(equity_curve_np)
    window = WINDOW_SIZE

    # Plot results (all advanced plots)
    plot_equity_curve(
        equity_curve,
        rewards=rewards,
        returns=returns,
        drawdown=drawdown,
        window=window
    )

    # Write trade statistics to CSV
    stats_csv_path = "plots/trade_statistics.csv"
    with open(stats_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        for key, value in trade_stats.items():
            writer.writerow([key, value])
    print(f"Trade statistics written to {stats_csv_path}")

    # Additional analysis and insights
    print("\n=== Detailed Analysis ===")

    # Analyze trade distribution
    if trades:
        # Trade duration analysis
        durations = [t['duration'] for t in trades]
        avg_duration = sum(durations) / len(durations)
        print(f"Average trade duration: {avg_duration:.1f} steps ({avg_duration / 288:.1f} days)")

        # Profit distribution
        profits = [t['net_pnl'] for t in trades]
        positive_profits = [p for p in profits if p > 0]
        negative_profits = [p for p in profits if p < 0]

        if positive_profits:
            print(f"Best winning trade: ${max(positive_profits):.2f}")
        if negative_profits:
            print(f"Worst losing trade: ${min(negative_profits):.2f}")

        # Commission impact
        total_commission = sum(t['commission'] for t in trades)
        total_gross_pnl = sum(t['pnl'] for t in trades)
        commission_impact = (total_commission / total_gross_pnl * 100) if total_gross_pnl != 0 else 0
        print(f"Commission impact: {commission_impact:.2f}% of gross PnL")

        # Position type performance
        long_trades = [t for t in trades if t['position_type'] == 'long']
        short_trades = [t for t in trades if t['position_type'] == 'short']

        if long_trades:
            long_pnl = sum(t['net_pnl'] for t in long_trades)
            print(f"Long trades total PnL: ${long_pnl:.2f}")

        if short_trades:
            short_pnl = sum(t['net_pnl'] for t in short_trades)
            print(f"Short trades total PnL: ${short_pnl:.2f}")

    print("\n=== Performance Summary ===")
    print(f"Strategy performance: {'PROFITABLE' if results['total_return'] > 0 else 'UNPROFITABLE'}")
    print(
        f"Risk-adjusted return (Sharpe): {'GOOD' if results['sharpe_ratio'] > 1.0 else 'POOR' if results['sharpe_ratio'] < 0 else 'MODERATE'}")
    print(
        f"Risk level (Max DD): {'HIGH' if results['max_drawdown'] > 0.2 else 'MODERATE' if results['max_drawdown'] > 0.1 else 'LOW'}")

    # Save comprehensive results to JSON for later analysis
    results_json_path = "plots/evaluation_results.json"

    # Prepare results for JSON serialization (convert numpy types to native Python types)
    json_results = {
        'performance_metrics': {
            'total_return': float(results['total_return']),
            'sharpe_ratio': float(results['sharpe_ratio']),
            'max_drawdown': float(results['max_drawdown']),
            'cumulative_return': float(equity_curve[-1] - equity_curve[0]) if equity_curve else 0.0
        },
        'trade_statistics': trade_stats,
        'configuration': {
            'window_size': WINDOW_SIZE,
            'max_episode_steps': MAX_EPISODE_STEPS,
            'total_timesteps': TOTAL_TIMESTEPS,
            'train_ratio': TRAIN_RATIO
        },
        'trade_count': len(trades),
        'evaluation_timestamp': str(pd.Timestamp.now())
    }

    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)

    with open(results_json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Comprehensive results saved to {results_json_path}")

    print(f"\n=== Files Generated ===")
    print(f"1. trade_history/trades.csv - Detailed trade data")
    print(f"2. {stats_csv_path} - Trade statistics")
    print(f"3. {results_json_path} - Complete evaluation results")
    print(f"4. plots/equity_curve.png - Equity curve visualization")
