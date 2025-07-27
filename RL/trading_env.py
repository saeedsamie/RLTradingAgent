import logging

import gymnasium as gym
import numpy as np
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

    def __init__(self, df, window_size=50, commission_per_lot=0.5, lot_size=0.01, debug=False,
                 max_episode_steps=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.current_step = window_size
        self.debug = debug
        self.max_episode_steps = max_episode_steps
        # First column is 'close', rest are features
        self.feature_cols = df.columns[1:]
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
        self.reset()

        if self.debug:
            logger.info(f"TradingEnv initialized with {len(self.df)} data points")
            logger.info(f"Window size: {self.window_size}, Max episode steps: {self.max_episode_steps}")
            logger.info(f"Action space: {self.action_space}")
            logger.info(f"Observation space: {self.observation_space}")
            logger.info(f"Initial balance: ${self.initial_balance}")

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.position = 0  # 0: Out, 1: Long, -1: Short
        self.entry_price = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.equity_curve = [self.equity]
        self.total_trades = 0
        self.total_commission = 0
        self.total_pnl = 0

        if self.debug:
            logger.info(f"Environment reset - Step: {self.current_step}, Balance: ${self.balance}")

        return self._get_obs(), {}

    def _get_obs(self):
        # Only use feature columns (not 'close') for observation
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step, 1:].values.astype(np.float32)
        # Add account state features to each row in the window
        balance_arr = np.full((self.window_size, 1), self.balance, dtype=np.float32)
        position_arr = np.full((self.window_size, 1), self.position, dtype=np.float32)
        obs = np.concatenate([obs, balance_arr, position_arr], axis=1)
        if np.isnan(obs).any() or np.isinf(obs).any():
            logger.warning(f"Observation contains NaN or inf at step {self.current_step}")
        return obs

    def step(self, action):
        # action: 0=Out, 1=Long, 2=Short
        if isinstance(action, (np.ndarray, list)):
            action = int(action[0]) if hasattr(action, '__len__') and len(action) > 0 else int(action)
        elif torch.is_tensor(action):
            action = int(action.item())
        else:
            # Handle scalar actions (integers)
            action = int(action)

        # Check episode termination conditions
        done = (self.current_step >= len(self.df) - 1) or (
                self.current_step >= self.window_size + self.max_episode_steps)

        reward = 0
        price = self.df.iloc[self.current_step]['close']
        commission = self.lot_size * self.commission_per_lot

        old_balance = self.balance
        old_position = self.position

        # Trading logic
        if action == 1:  # Long
            if self.position == 0:
                self.entry_price = price
                self.position = 1
                self.balance -= commission
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN LONG - Price: ${price:.2f}, Lot: {self.lot_size:.2f}, Commission: ${commission:.2f}")
            elif self.position == -1:
                pnl = (self.entry_price - price) * self.lot_size * 10000  # pip value
                reward = pnl - commission
                self.balance += reward
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
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN LONG - Price: ${price:.2f}, Lot: {self.lot_size:.2f}, Commission: ${commission:.2f}")

        elif action == 2:  # Short
            if self.position == 0:
                self.entry_price = price
                self.position = -1
                self.balance -= commission
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN SHORT - Price: ${price:.2f}, Lot: {self.lot_size:.2f}, Commission: ${commission:.2f}")
            elif self.position == 1:
                pnl = (price - self.entry_price) * self.lot_size * 10000  # pip value
                reward = pnl - commission
                self.balance += reward
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
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN SHORT - Price: ${price:.2f}, Lot: {self.lot_size:.2f}, Commission: ${commission:.2f}")

        elif action == 0:  # Out
            if self.position == 1:
                pnl = (price - self.entry_price) * self.lot_size * 10000  # pip value
                reward = pnl - commission
                self.balance += reward
                self.position = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE LONG - Price: ${price:.2f}, PnL: ${pnl:.2f}, Reward: ${reward:.2f}")
            elif self.position == -1:
                pnl = (self.entry_price - price) * self.lot_size * 10000  # pip value
                reward = pnl - commission
                self.balance += reward
                self.position = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE SHORT - Price: ${price:.2f}, PnL: ${pnl:.2f}, Reward: ${reward:.2f}")

        # Update equity
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (price - self.entry_price) * self.lot_size * 10000
            else:  # position == -1
                unrealized_pnl = (self.entry_price - price) * self.lot_size * 10000
            self.equity = self.balance + unrealized_pnl
        else:
            self.equity = self.balance

        self.equity_curve.append(self.equity)
        self.current_step += 1

        # Add small penalty for holding positions to encourage trading
        if self.position != 0:
            reward -= 0.001

        # Calculate unrealized PnL
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = (price - self.entry_price) * self.lot_size * 10000
            else:  # position == -1
                unrealized_pnl = (self.entry_price - price) * self.lot_size * 10000
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

        return self._get_obs(), reward, done, False, info

    def render(self):
        pass
