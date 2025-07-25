import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch

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

    def __init__(self, df, window_size=50, commission_per_lot=0.5, min_lot=0.01, max_lot=1.0):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.current_step = window_size
        self.num_features = df.shape[1] - 1
        # Action: (0=Out, 1=Long, 2=Short), Confidence (0-1)
        self.action_space = spaces.Tuple([
            spaces.Discrete(3),
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        ])
        # +3 for [balance, position, lot_size]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, self.num_features + 3), dtype=np.float32
        )
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.commission_per_lot = commission_per_lot
        self.initial_balance = 1000.0
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.position = 0  # 0: Out, 1: Long, -1: Short
        self.entry_price = 0
        self.lot_size = 0
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.equity_curve = [self.equity]
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step, 1:].values.astype(np.float32)
        # Add account state features to each row in the window
        balance_arr = np.full((self.window_size, 1), self.balance, dtype=np.float32)
        position_arr = np.full((self.window_size, 1), self.position, dtype=np.float32)
        lot_size_arr = np.full((self.window_size, 1), self.lot_size, dtype=np.float32)
        obs = np.concatenate([obs, balance_arr, position_arr, lot_size_arr], axis=1)
        if np.isnan(obs).any() or np.isinf(obs).any():
            print(f"[WARNING] Observation contains NaN or inf at step {self.current_step}")
        return obs

    def step(self, action):
        if isinstance(action, (np.ndarray, list)) and len(action) == 2:
            act = int(action[0])
            conf = float(action[1])
            action = (act, conf)
        elif torch.is_tensor(action) and action.numel() == 2:
            act = int(action[0].item())
            conf = float(action[1].item())
            action = (act, conf)
        done = self.current_step >= len(self.df) - 1
        reward = 0
        price = self.df.iloc[self.current_step]['close']
        act, conf = action
        conf = float(np.clip(conf, 0, 1))
        lot = self.min_lot + (self.max_lot - self.min_lot) * conf
        commission = lot * self.commission_per_lot
        # Trading logic
        if act == 1:  # Long
            if self.position == 0:
                self.entry_price = price
                self.position = 1
                self.lot_size = lot
                self.balance -= commission
            elif self.position == -1:
                pnl = (self.entry_price - price) * self.lot_size * 10000  # pip value
                reward = pnl - commission
                self.balance += reward
                self.position = 0
                self.lot_size = 0
        elif act == 2:  # Short
            if self.position == 0:
                self.entry_price = price
                self.position = -1
                self.lot_size = lot
                self.balance -= commission
            elif self.position == 1:
                pnl = (price - self.entry_price) * self.lot_size * 10000
                reward = pnl - commission
                self.balance += reward
                self.position = 0
                self.lot_size = 0
        else:  # Out
            if self.position == 1:
                pnl = (price - self.entry_price) * self.lot_size * 10000
                reward = pnl - commission
                self.balance += reward
                self.position = 0
                self.lot_size = 0
            elif self.position == -1:
                pnl = (self.entry_price - price) * self.lot_size * 10000
                reward = pnl - commission
                self.balance += reward
                self.position = 0
                self.lot_size = 0
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
            print(f"[WARNING] Reward is NaN or inf at step {self.current_step}")
        self.current_step += 1
        obs = self._get_obs()
        return obs, reward, done, False, {}

    def render(self):
        print(f'Step: {self.current_step}, Equity: {self.equity:.2f}, Balance: {self.balance:.2f}, Position: {self.position}, Lot: {self.lot_size:.2f}')
