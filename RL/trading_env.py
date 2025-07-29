import logging
import time

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
        start_time = time.time()
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

        result = self._get_obs(), {}
        elapsed = time.time() - start_time
        if not hasattr(self, 'reset_times'):
            self.reset_times = []
        self.reset_times.append(elapsed)
        return result

    def _get_obs(self):
        # Only use feature columns (not 'close') for observation
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        # Ensure we have enough data
        if end_idx > len(self.df):
            raise ValueError(f"Current step {self.current_step} exceeds dataframe length {len(self.df)}")
        
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

        # Improved reward function to prevent overfitting
        if self.current_step > self.window_size:
            # Calculate risk-adjusted reward
            if len(self.equity_curve) > 1:
                # Calculate rolling volatility (risk measure)
                recent_equity = self.equity_curve[-min(50, len(self.equity_curve)):]
                if len(recent_equity) > 1:
                    returns = [(recent_equity[i] - recent_equity[i-1]) / recent_equity[i-1] 
                              for i in range(1, len(recent_equity))]
                    volatility = np.std(returns) if returns else 0.01
                    
                    # Risk-adjusted reward (Sharpe-like)
                    if volatility > 0:
                        risk_adjusted_reward = reward / (volatility + 0.01)
                    else:
                        risk_adjusted_reward = reward
                else:
                    risk_adjusted_reward = reward
            else:
                risk_adjusted_reward = reward
            
            # Add drawdown penalty
            if len(self.equity_curve) > 1:
                peak_equity = max(self.equity_curve)
                current_drawdown = (peak_equity - self.equity) / peak_equity if peak_equity > 0 else 0
                drawdown_penalty = -current_drawdown * 0.1  # Penalize drawdowns
            else:
                drawdown_penalty = 0
            
            # Add diversity penalty to encourage exploration
            if hasattr(self, 'action_history'):
                if len(self.action_history) > 10:
                    recent_actions = self.action_history[-10:]
                    action_diversity = len(set(recent_actions)) / 3.0  # 3 possible actions
                    diversity_bonus = (action_diversity - 0.5) * 0.01  # Encourage diverse actions
                else:
                    diversity_bonus = 0
            else:
                diversity_bonus = 0
                self.action_history = []
            
            # Combine all reward components
            final_reward = risk_adjusted_reward + drawdown_penalty + diversity_bonus
            
            # Store action for diversity tracking
            self.action_history.append(action)
            
            # Cap extreme rewards to prevent overfitting
            final_reward = np.clip(final_reward, -100, 100)
        else:
            final_reward = reward

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

        result = self._get_obs(), final_reward, done, False, info
        elapsed = time.time() - start_time
        if not hasattr(self, 'step_times'):
            self.step_times = []
        self.step_times.append(elapsed)
        return result

    def render(self):
        pass
