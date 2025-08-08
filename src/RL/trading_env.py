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
    - Uses Differential Sharpe Ratio (DSR) for reward calculation
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=288, commission_per_lot=50, lot_size=0.01, debug=False,
                 max_episode_steps=25920, beta=0.9, alpha=5.0, epsilon=1e-6, initial_balance=1000.0):
        """
        Initialize the trading environment with DSR-based reward function.

        Args:
            df (pd.DataFrame): DataFrame with price data and features. First column must be 'close'.
            window_size (int): Number of time steps to include in each observation.
            commission_per_lot (float): Commission fee per lot traded.
            lot_size (float): Size of each trading lot.
            debug (bool): Whether to enable debug logging.
            max_episode_steps (int): Maximum number of steps per episode.
            beta (float): EMA decay factor for DSR calculation (0.9-0.99).
                Higher values give more weight to historical data.
            alpha (float): Penalty weight for spread costs in reward calculation.
                Higher values discourage frequent trading.
            epsilon (float): Small constant to avoid division by zero in Sharpe calculation.
            initial_balance (float): Initial account balance in USD.
        """
        super().__init__()

        # Validate and store input parameters
        if not 0 < beta < 1:
            raise ValueError("beta must be between 0 and 1")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive")

        # Store DataFrame with datetime index
        self.datetime_index = df.index.copy()
        self.df = df.reset_index(drop=True)

        # Basic environment parameters
        self.window_size = window_size
        self.current_step = window_size
        self.debug = debug
        self.max_episode_steps = max_episode_steps
        self.initial_balance = initial_balance

        # Trading parameters
        self.lot_size = lot_size
        self.commission_per_lot = commission_per_lot

        # Feature setup
        self.feature_cols = self.df.columns[1:]  # First column is 'close'
        self.num_features = len(self.feature_cols)

        # Action and observation space
        self.action_space = spaces.Discrete(3)  # 0=Out, 1=Long, 2=Short
        self.observation_space = spaces.Box(  # Features + [balance, position]
            low=-np.inf, high=np.inf, 
            shape=(window_size, self.num_features + 2), 
            dtype=np.float32
        )

        # Episode window setup
        self.episode_start_idx = 0
        self.episode_end_idx = min(window_size + max_episode_steps, len(self.df))

        # DSR parameters
        self.beta = beta  # EMA decay factor (0.9-0.99)
        self.alpha = alpha  # Spread penalty weight
        self.epsilon = epsilon  # Small constant to avoid division by zero

        # Initialize DSR variables
        self._init_dsr_variables()

        if self.debug:
            self._log_initialization()

        self.reset()

    def _init_dsr_variables(self):
        """Initialize or reset DSR-related variables."""
        self.mu = 0  # EMA of returns
        self.sigma2 = self.epsilon  # EMA of squared deviations (variance)
        self.prev_sharpe = 0  # Previous Sharpe ratio
        self.returns_history = []  # List to store returns for analysis
        self.current_return = 0  # Current return
        self.current_sharpe = 0  # Current Sharpe ratio

    def _log_initialization(self):
        """Log initialization parameters when debug is enabled."""
        logger.info(f"TradingEnv initialized with {len(self.df)} data points")
        logger.info(f"Window size: {self.window_size}, Max episode steps: {self.max_episode_steps}")
        logger.info(f"Action space: {self.action_space}")
        logger.info(f"Observation space: {self.observation_space}")
        logger.info(f"Initial balance: ${self.initial_balance}")
        logger.info(f"Datetime range: {self.datetime_index[0]} to {self.datetime_index[-1]}")
        logger.info(f"Episode range: {self.episode_start_idx} to {self.episode_end_idx}")
        logger.info(f"DSR parameters - beta: {self.beta}, alpha: {self.alpha}, epsilon: {self.epsilon}")
        logger.info(f"Trading parameters - Lot size: {self.lot_size}, Commission per lot: {self.commission_per_lot}")

    def _calculate_return(self, old_balance, new_balance):
        """Calculate return for DSR computation"""
        if old_balance == 0:
            return 0
        return (new_balance - old_balance) / old_balance

    def _update_dsr_stats(self, return_t):
        """Update DSR statistics with new return"""
        # Update EMA of returns
        self.mu = self.beta * self.mu + (1 - self.beta) * return_t
        
        # Update EMA of squared deviations (variance)
        self.sigma2 = self.beta * self.sigma2 + (1 - self.beta) * (return_t - self.mu) ** 2
        
        # Calculate current Sharpe ratio
        current_sharpe = self.mu / (np.sqrt(self.sigma2) + self.epsilon)
        
        # Calculate DSR (change in Sharpe)
        dsr = current_sharpe - self.prev_sharpe
        
        # Store current Sharpe as previous for next step
        self.prev_sharpe = current_sharpe
        
        return dsr

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
        self.trades_this_episode = 0  # Reset trades counter for new episode
        self.entry_datetime = None  # Initialize entry datetime tracking

        # Reset DSR-related variables
        self.mu = 0  # Reset EMA of returns
        self.sigma2 = self.epsilon  # Reset EMA of squared deviations (variance)
        self.prev_sharpe = 0  # Reset previous Sharpe ratio
        self.returns_history = []  # Reset returns history

        if self.debug:
            logger.info(
                f"Environment reset - Episode: {self.episode_start_idx} to {self.episode_end_idx}, Step: {self.current_step}, Balance: ${self.balance}")

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

        # DSR-based reward function with spread penalty
        reward = 0
        price = self.df.iloc[self.current_step]['close']
        spread = self.lot_size * self.commission_per_lot / 2  # Using commission as spread for now

        old_balance = self.balance
        old_position = self.position
        old_equity = self.equity

        # Track trades per episode
        if not hasattr(self, 'trades_this_episode'):
            self.trades_this_episode = 0

        # Trading logic with IMPROVED rewards
        if action == 1:  # Long
            if self.position == 0:
                # Opening new long position
                self.entry_price = price
                self.position = 1
                self.balance -= spread
                self.total_commission += spread
                self.trades_this_episode += 1

                # Calculate return and update DSR
                new_equity = self.balance  # No unrealized PnL yet
                return_t = self._calculate_return(old_equity, new_equity)
                self.returns_history.append(return_t)
                dsr = self._update_dsr_stats(return_t)
                
                # DSR reward with spread penalty
                reward = dsr - self.alpha * spread

                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN LONG - Price: ${price:.2f}, DSR: {dsr:.4f}, Spread Penalty: {self.alpha * spread:.4f}")

            elif self.position == -1:
                # Closing short and opening long
                pnl = (self.entry_price - price) * self.lot_size * 100
                self.balance += pnl - spread  # Close short position
                self.position = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += spread

                # Calculate return and update DSR for closing short
                new_equity = self.balance
                return_t = self._calculate_return(old_equity, new_equity)
                self.returns_history.append(return_t)
                dsr = self._update_dsr_stats(return_t)

                # Open new long position
                self.entry_price = price
                self.position = 1
                self.balance -= spread  # Open long position
                self.total_commission += spread
                self.trades_this_episode += 1

                # Calculate final DSR reward with spread penalty
                reward = dsr - self.alpha * (2 * spread)  # Double spread for two actions

                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE SHORT & OPEN LONG - PnL: ${pnl:.2f}, DSR: {dsr:.4f}, Spread Penalty: {self.alpha * 2 * spread:.4f}")

        elif action == 2:  # Short
            if self.position == 0:
                # Opening new short position
                self.entry_price = price
                self.position = -1
                self.balance -= spread
                self.total_commission += spread
                self.trades_this_episode += 1

                # Calculate return and update DSR
                new_equity = self.balance  # No unrealized PnL yet
                return_t = self._calculate_return(old_equity, new_equity)
                self.returns_history.append(return_t)
                dsr = self._update_dsr_stats(return_t)
                
                # DSR reward with spread penalty
                reward = dsr - self.alpha * spread

                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: OPEN SHORT - Price: ${price:.2f}, DSR: {dsr:.4f}, Spread Penalty: {self.alpha * spread:.4f}")

            elif self.position == 1:
                # Closing long and opening short
                pnl = (price - self.entry_price) * self.lot_size * 100
                self.balance += pnl - spread  # Close long position
                self.position = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += spread

                # Calculate return and update DSR for closing long
                new_equity = self.balance
                return_t = self._calculate_return(old_equity, new_equity)
                self.returns_history.append(return_t)
                dsr = self._update_dsr_stats(return_t)

                # Open new short position
                self.entry_price = price
                self.position = -1
                self.balance -= spread  # Open short position
                self.total_commission += spread
                self.trades_this_episode += 1

                # Calculate final DSR reward with spread penalty
                reward = dsr - self.alpha * (2 * spread)  # Double spread for two actions

                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE LONG & OPEN SHORT - PnL: ${pnl:.2f}, DSR: {dsr:.4f}, Spread Penalty: {self.alpha * 2 * spread:.4f}")

        elif action == 0:  # Out
            if self.position == 1:
                # Closing long position
                pnl = (price - self.entry_price) * self.lot_size * 100
                self.balance += pnl - spread
                self.position = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += spread

                # Calculate return and update DSR
                new_equity = self.balance
                return_t = self._calculate_return(old_equity, new_equity)
                self.returns_history.append(return_t)
                dsr = self._update_dsr_stats(return_t)
                
                # DSR reward with spread penalty
                reward = dsr - self.alpha * spread

                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE LONG - PnL: ${pnl:.2f}, DSR: {dsr:.4f}, Spread Penalty: {self.alpha * spread:.4f}")

            elif self.position == -1:
                # Closing short position
                pnl = (self.entry_price - price) * self.lot_size * 100
                self.balance += pnl - spread
                self.position = 0
                self.total_trades += 1
                self.total_pnl += pnl
                self.total_commission += spread

                # Calculate return and update DSR
                new_equity = self.balance
                return_t = self._calculate_return(old_equity, new_equity)
                self.returns_history.append(return_t)
                dsr = self._update_dsr_stats(return_t)
                
                # DSR reward with spread penalty
                reward = dsr - self.alpha * spread

                if self.debug:
                    logger.info(
                        f"Step {self.current_step}: CLOSE SHORT - PnL: ${pnl:.2f}, DSR: {dsr:.4f}, Spread Penalty: {self.alpha * spread:.4f}")

            else:
                # No position and choosing to stay out - Calculate DSR on current equity
                new_equity = self.balance
                return_t = self._calculate_return(old_equity, new_equity)
                self.returns_history.append(return_t)
                dsr = self._update_dsr_stats(return_t)
                reward = dsr  # No spread penalty when staying out

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
                'commission': spread,
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
            'spread': spread,
            # DSR-related metrics
            'dsr_mu': self.mu,
            'dsr_sigma': np.sqrt(self.sigma2),
            'current_sharpe': self.mu / (np.sqrt(self.sigma2) + self.epsilon),
            'prev_sharpe': self.prev_sharpe,
            'last_return': self.returns_history[-1] if self.returns_history else 0.0
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
