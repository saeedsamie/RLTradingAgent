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
    - Confidence (0-1) determines lot size (min 0.01 lots)
    - Commission fee per contract
    - Tracks USD balance, open positions, and PnL
    - Observation includes windowed features + [balance, position, lot_size]
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=50, commission_per_lot=0.5, min_lot=0.01, max_lot=1.0, debug=False,
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
        # Action: [action_type, confidence] where action_type in [0,2], confidence in [0,1]
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([2, 1]), dtype=np.float32)
        # +3 for [balance, position, lot_size]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, self.num_features + 3), dtype=np.float32
        )
        self.min_lot = min_lot
        self.max_lot = max_lot
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
        self.lot_size = 0
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
        lot_size_arr = np.full((self.window_size, 1), self.lot_size, dtype=np.float32)
        obs = np.concatenate([obs, balance_arr, position_arr, lot_size_arr], axis=1)
        if np.isnan(obs).any() or np.isinf(obs).any():
            logger.warning(f"Observation contains NaN or inf at step {self.current_step}")
        return obs

    def step(self, action):
        # action: [action_type, confidence]
        if isinstance(action, (np.ndarray, list)) and len(action) == 2:
            act = int(np.round(action[0]))
            conf = float(np.clip(action[1], 0, 1))
            action = (act, conf)
        elif torch.is_tensor(action) and action.numel() == 2:
            act = int(torch.round(action[0]).item())
            conf = float(torch.clamp(action[1], 0, 1).item())
            action = (act, conf)

        # Check episode termination conditions
        done = (self.current_step >= len(self.df) - 1) or (
                self.current_step >= self.window_size + self.max_episode_steps)

        reward = 0
        price = self.df.iloc[self.current_step]['close']
        act, conf = action
        conf = float(np.clip(conf, 0, 1))
        lot = self.min_lot + (self.max_lot - self.min_lot) * conf
        commission = lot * self.commission_per_lot

        old_balance = self.balance
        old_position = self.position

        # Trading logic
        if act == 1:  # Long
            if self.position == 0:
                self.entry_price = price
                self.position = 1
                self.lot_size = lot
                self.balance -= commission
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN LONG - Price: ${price:.2f}, Lot: {lot:.2f}, Commission: ${commission:.2f}")
            elif self.position == -1:
                pnl = (self.entry_price - price) * self.lot_size * 10000  # pip value
                reward = pnl - commission
                self.balance += reward
                self.position = 0
                self.lot_size = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE SHORT - Price: ${price:.2f}, PnL: ${pnl:.2f}, Reward: ${reward:.2f}")
        elif act == 2:  # Short
            if self.position == 0:
                self.entry_price = price
                self.position = -1
                self.lot_size = lot
                self.balance -= commission
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN SHORT - Price: ${price:.2f}, Lot: {lot:.2f}, Commission: ${commission:.2f}")
            elif self.position == 1:
                pnl = (price - self.entry_price) * self.lot_size * 10000
                reward = pnl - commission
                self.balance += reward
                self.position = 0
                self.lot_size = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE LONG - Price: ${price:.2f}, PnL: ${pnl:.2f}, Reward: ${reward:.2f}")
        else:  # Out
            if self.position == 1:
                pnl = (price - self.entry_price) * self.lot_size * 10000
                reward = pnl - commission
                self.balance += reward
                self.position = 0
                self.lot_size = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE LONG (OUT) - Price: ${price:.2f}, PnL: ${pnl:.2f}, Reward: ${reward:.2f}")
            elif self.position == -1:
                pnl = (self.entry_price - price) * self.lot_size * 10000
                reward = pnl - commission
                self.balance += reward
                self.position = 0
                self.lot_size = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += commission
                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE SHORT (OUT) - Price: ${price:.2f}, PnL: ${pnl:.2f}, Reward: ${reward:.2f}")

        # Update equity
        if self.position == 1:
            floating_pnl = (price - self.entry_price) * self.lot_size * 10000
        elif self.position == -1:
            floating_pnl = (self.entry_price - price) * self.lot_size * 10000
        else:
            floating_pnl = 0
        self.equity = self.balance + floating_pnl
        self.equity_curve.append(self.equity)

        if np.isnan(reward) or np.isinf(reward):
            logger.warning(f"Reward is NaN or inf at step {self.current_step}")

        # Log every 1000 steps or when there's a significant change
        if self.debug and (self.current_step % 1000 == 0 or abs(reward) > 10 or self.position != old_position):
            logger.info(f"Step {self.current_step}: Action={action}, Position={self.position}, "
                        f"Balance=${self.balance:.2f}, Equity=${self.equity:.2f}, Reward=${reward:.2f}")

        self.current_step += 1
        obs = self._get_obs()

        # Log episode summary when done
        if done and self.debug:
            episode_length = self.current_step - self.window_size
            logger.info(f"Episode finished - Total steps: {episode_length}")
            logger.info(f"Final balance: ${self.balance:.2f}, Final equity: ${self.equity:.2f}")
            logger.info(f"Total trades: {self.total_trades}, Total PnL: ${self.total_pnl:.2f}")
            logger.info(f"Total commission: ${self.total_commission:.2f}")
            logger.info(f"Net profit: ${self.balance - self.initial_balance:.2f}")

        return obs, reward, done, False, {}

    def render(self):
        print(
            f'Step: {self.current_step}, Equity: {self.equity:.2f}, Balance: {self.balance:.2f}, Position: {self.position}, Lot: {self.lot_size:.2f}')
