import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """
    Custom trading environment for RL with actions:
    0: Out, 1: Long, 2: Short
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=50):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.current_step = window_size
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, df.shape[1] - 1), dtype=np.float32
        )
        self.position = 0  # 0: Out, 1: Long, -1: Short
        self.entry_price = 0
        self.equity = 1.0
        self.equity_curve = []

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0
        self.equity = 1.0
        self.equity_curve = [self.equity]
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.df.iloc[self.current_step - self.window_size:self.current_step, 1:].values
        return obs.astype(np.float32)

    def step(self, action):
        done = self.current_step >= len(self.df) - 1
        reward = 0
        price = self.df.iloc[self.current_step]['close']
        if action == 1:  # Long
            if self.position == 0:
                self.entry_price = price
                self.position = 1
            elif self.position == -1:
                reward = self.entry_price - price
                self.position = 0
        elif action == 2:  # Short
            if self.position == 0:
                self.entry_price = price
                self.position = -1
            elif self.position == 1:
                reward = price - self.entry_price
                self.position = 0
        else:  # Out
            if self.position == 1:
                reward = price - self.entry_price
                self.position = 0
            elif self.position == -1:
                reward = self.entry_price - price
                self.position = 0
        self.equity += reward
        self.equity_curve.append(self.equity)
        self.current_step += 1
        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(f'Step: {self.current_step}, Equity: {self.equity}, Position: {self.position}')
