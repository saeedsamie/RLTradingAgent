import logging
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Forex trading environment for RL:
    - Start with 1000 USD
    - Actions: 0=Out, 1=Long, 2=Short
    - Fixed lot size (0.01 lots)
    - Commission fee per contract
    - Tracks USD balance, open positions, and PnL
    - Observation includes windowed features + [balance, position]
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=288, commission_per_lot=50, lot_size=0.01, debug=False,
                 max_episode_steps=25920):
        super().__init__()
        # Preserve datetime information if available
        
        # DataFrame already has datetime index
        self.datetime_index = df.index.copy()
        self.df = df.reset_index(drop=True)
        
        self.window_size = window_size
        self.current_step = window_size
        self.debug = debug
        self.max_episode_steps = max_episode_steps
        # First column is 'close', rest are features
        self.feature_cols = self.df.columns[1:]
        self.num_features = len(self.feature_cols)
        # Action: 0=Out, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        # +2 for [balance, position]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, self.num_features + 2), dtype=np.float32
        )
        self.lot_size = lot_size
        self.commission_per_lot = commission_per_lot
        self.initial_balance = 1000.0
        
        # Add sliding window support for multiple episodes
        self.episode_start_idx = 0
        self.episode_end_idx = min(window_size + max_episode_steps, len(self.df))
        
        self.reset()

        if self.debug:
            logger.info(f"TradingEnv initialized with {len(self.df)} data points")
            logger.info(f"Window size: {self.window_size}, Max episode steps: {self.max_episode_steps}")
            logger.info(f"Action space: {self.action_space}")
            logger.info(f"Observation space: {self.observation_space}")
            logger.info(f"Initial balance: ${self.initial_balance}")
            logger.info(f"Datetime range: {self.datetime_index[0]} to {self.datetime_index[-1]}")
            logger.info(f"Episode range: {self.episode_start_idx} to {self.episode_end_idx}")

    def get_datetime_at_index(self, index):
        """
        Get datetime for a specific index in the dataset.
        
        Args:
            index (int): The index to get datetime for
            
        Returns:
            datetime or None: The datetime at the given index, or None if index is out of bounds
        """
        if index is None or index < 0 or index >= len(self.datetime_index):
            return None
        return self.datetime_index[index]

    def reset(self, seed=None, options=None):
        start_time = time.time()
        
        # Implement sliding window for multiple episodes
        # Move the episode window forward by max_episode_steps, but ensure we have enough data
        if self.episode_end_idx >= len(self.df) - self.max_episode_steps:
            # If we're near the end, start over from the beginning
            self.episode_start_idx = 0
            self.episode_end_idx = min(self.window_size + self.max_episode_steps, len(self.df))
        else:
            # Move the window forward
            self.episode_start_idx = self.episode_end_idx - self.max_episode_steps
            self.episode_end_idx = min(self.episode_start_idx + self.window_size + self.max_episode_steps, len(self.df))
        
        # Set current step to the start of the episode window
        self.current_step = self.episode_start_idx + self.window_size
        
        self.position = 0  # 0: Out, 1: Long, -1: Short
        self.entry_price = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.equity_curve = [self.equity]
        self.total_trades = 0
        self.total_commission = 0
        self.total_pnl = 0
        self.entry_step = None  # Initialize entry step tracking
        self.entry_datetime = None  # Initialize entry datetime tracking

        if self.debug:
            logger.info(f"Environment reset - Episode: {self.episode_start_idx} to {self.episode_end_idx}, Step: {self.current_step}, Balance: ${self.balance}")

        result = self._get_obs(), {}
        elapsed = time.time() - start_time
        if not hasattr(self, 'reset_times'):
            self.reset_times = []
        self.reset_times.append(elapsed)
        return result

    def _get_obs(self):
        # CRITICAL FIX: Use data up to previous step to prevent look-ahead bias
        # Use sliding window approach - ensure we stay within episode bounds
        start_idx = max(self.episode_start_idx, self.current_step - self.window_size)
        end_idx = min(self.current_step - 1, self.episode_end_idx)  # Use data up to previous step only

        # Ensure we have enough data
        if end_idx > len(self.df):
            raise ValueError(f"Current step {self.current_step} exceeds dataframe length {len(self.df)}")

        # If we don't have enough historical data, pad with the earliest available data in episode
        if end_idx < start_idx:
            end_idx = start_idx
        
        obs = self.df.iloc[start_idx:end_idx, 1:].values.astype(np.float32)

        # If we don't have enough data for the full window, pad with zeros
        if obs.shape[0] < self.window_size:
            padding_rows = self.window_size - obs.shape[0]
            padding = np.zeros((padding_rows, obs.shape[1]), dtype=np.float32)
            obs = np.vstack([padding, obs])

        # Add account state features to each row in the window
        balance_arr = np.full((self.window_size, 1), self.balance, dtype=np.float32)
        position_arr = np.full((self.window_size, 1), self.position, dtype=np.float32)
        obs = np.concatenate([obs, balance_arr, position_arr], axis=1)

        # Ensure obs is a numpy array before checking for NaN/inf
        if isinstance(obs, np.ndarray) and (np.isnan(obs).any() or np.isinf(obs).any()):
            logger.warning(f"Observation contains NaN or inf at step {self.current_step}")

        return obs

    def step(self, action):
        start_time = time.time()
        # action: 0=Out, 1=Long, 2=Short
        try:
            if isinstance(action, np.ndarray):
                if action.size > 0:
                    action = int(action.flat[0])
                else:
                    action = int(action)
            elif isinstance(action, list):
                if len(action) > 0:
                    action = int(action[0])
                else:
                    action = int(action)
            elif torch.is_tensor(action):
                action = int(action.item())
            else:
                # Handle scalar actions (integers)
                action = int(action)
        except (TypeError, ValueError, IndexError) as e:
            logger.warning(f"Error processing action {action} of type {type(action)}: {e}")
            action = 0  # Default to 'Out' action

        # Check episode termination conditions - use sliding window approach
        # Episode ends when we reach the end of the current episode window
        done = (self.current_step >= self.episode_end_idx - 1) or (
                self.current_step >= self.episode_start_idx + self.window_size + self.max_episode_steps)

        reward = 0
        price = self.df.iloc[self.current_step]['close']
        commission = self.lot_size * self.commission_per_lot

        old_balance = self.balance
        old_position = self.position

        # Trading logic with realistic rewards
        if action == 1:  # Long
            if self.position == 0:
                self.entry_price = price
                self.position = 1
                self.balance -= commission
                self.total_commission += commission
                reward = 0.0  # No reward for taking action
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN LONG - Price: ${price:.2f}, Lot: {self.lot_size:.2f}, Commission: ${commission:.2f}")
            elif self.position == -1:
                pnl = (self.entry_price - price) * self.lot_size * 100  # Realistic pip value
                reward = pnl - commission  # No artificial bonus
                self.balance += pnl - commission
                self.position = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE SHORT - Price: ${price:.2f}, PnL: ${pnl:.2f}, Reward: ${reward:.2f}")
                # Open new long position
                self.entry_price = price
                self.position = 1
                self.balance -= commission
                self.total_commission += commission
                # No additional reward for new position
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN LONG - Price: ${price:.2f}, Lot: {self.lot_size:.2f}, Commission: ${commission:.2f}")

        elif action == 2:  # Short
            if self.position == 0:
                self.entry_price = price
                self.position = -1
                self.balance -= commission
                self.total_commission += commission
                reward = 0.0  # No reward for taking action
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN SHORT - Price: ${price:.2f}, Lot: {self.lot_size:.2f}, Commission: ${commission:.2f}")
            elif self.position == 1:
                pnl = (price - self.entry_price) * self.lot_size * 100  # Realistic pip value
                reward = pnl - commission  # No artificial bonus
                self.balance += pnl - commission
                self.position = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE LONG - Price: ${price:.2f}, PnL: ${pnl:.2f}, Reward: ${reward:.2f}")
                # Open new short position
                self.entry_price = price
                self.position = -1
                self.balance -= commission
                self.total_commission += commission
                # No additional reward for new position
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN SHORT - Price: ${price:.2f}, Lot: {self.lot_size:.2f}, Commission: ${commission:.2f}")

        elif action == 0:  # Out
            if self.position == 1:
                pnl = (price - self.entry_price) * self.lot_size * 100  # Realistic pip value
                reward = pnl - commission  # No artificial bonus
                self.balance += pnl - commission
                self.position = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE LONG - Price: ${price:.2f}, PnL: ${pnl:.2f}, Reward: ${reward:.2f}")
            elif self.position == -1:
                pnl = (self.entry_price - price) * self.lot_size * 100  # Realistic pip value
                reward = pnl - commission  # No artificial bonus
                self.balance += pnl - commission
                self.position = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE SHORT - Price: ${price:.2f}, PnL: ${pnl:.2f}, Reward: ${reward:.2f}")
            else:
                # No position and choosing to stay out - no penalty (realistic)
                reward = 0.0

        # Update equity
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (price - self.entry_price) * self.lot_size * 100
            else:  # position == -1
                unrealized_pnl = (self.entry_price - price) * self.lot_size * 100
            self.equity = self.balance + unrealized_pnl
        else:
            self.equity = self.balance

        self.equity_curve.append(self.equity)

        # Track actual trades for proper analysis
        if not hasattr(self, 'trade_history'):
            self.trade_history = []

        # Record trade if position was closed
        if old_position != 0 and self.position == 0:
            # Position was closed - record the trade
            # Calculate PnL for this specific trade
            if old_position == 1:  # Long position
                trade_pnl = (price - self.entry_price) * self.lot_size * 100
            else:  # Short position
                trade_pnl = (self.entry_price - price) * self.lot_size * 100

            # Get datetime information using the centralized function
            entry_datetime = self.get_datetime_at_index(self.entry_step)
            exit_datetime = self.get_datetime_at_index(self.current_step)

            trade_info = {
                'entry_idx': self.entry_step if hasattr(self, 'entry_step') else None,
                'exit_idx': self.current_step,
                'entry_datetime': entry_datetime,
                'exit_datetime': exit_datetime,
                'entry_price': self.entry_price,
                'exit_price': price,
                'position_type': 'long' if old_position == 1 else 'short',
                'pnl': trade_pnl,
                'commission': commission,
                'duration': self.current_step - self.entry_step if hasattr(self, 'entry_step') else 0,
                'reward': reward,
                'balance': self.balance,
                'equity': self.equity,
                'lot_size': self.lot_size
            }
            self.trade_history.append(trade_info)

        # Record trade entry if position was opened
        if old_position == 0 and self.position != 0:
            self.entry_step = self.current_step
            self.entry_datetime = self.get_datetime_at_index(self.current_step)
            if self.debug:
                logger.info(f"Trade entry recorded at step {self.current_step} at {self.entry_datetime}")

        self.current_step += 1

        # Simplified reward function - just use the basic reward
        final_reward = reward

        # Calculate unrealized PnL
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (price - self.entry_price) * self.lot_size * 100
            else:  # position == -1
                unrealized_pnl = (self.entry_price - price) * self.lot_size * 100
        else:
            unrealized_pnl = 0.0

        info = {
            'balance': self.balance,
            'equity': self.equity,
            'position': self.position,
            'total_trades': self.total_trades,
            'total_commission': self.total_commission,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'price': price,
            'commission': commission
        }

        result = self._get_obs(), final_reward, done, False, info
        elapsed = time.time() - start_time
        if not hasattr(self, 'step_times'):
            self.step_times = []
        self.step_times.append(elapsed)
        # Write trade history to a file with a unique name at the end of each episode
        if done:
            import os
            import uuid
            if hasattr(self, 'trade_history') and len(self.trade_history) > 0:
                unique_id = uuid.uuid4().hex
                filename = f"trade_history.csv"
                output_dir = "trade_history"
                os.makedirs(output_dir, exist_ok=True)
                filepath = os.path.join(output_dir, filename)
                try:
                    import csv
                    # Get all possible keys from all trades for CSV header
                    all_keys = set()
                    for trade in self.trade_history:
                        all_keys.update(trade.keys())
                    fieldnames = sorted(list(all_keys))
                    with open(filepath, "w", newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for trade in self.trade_history:
                            writer.writerow(trade)
                    if self.debug:
                        logger.info(f"Trade history written to {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to write trade history to file: {e}")
        return result

    def get_datetime_at_step(self, step):
        """Get datetime at a specific step"""
        if step < len(self.datetime_index):
            return self.datetime_index[step]
        return None

    def render(self):
        pass
