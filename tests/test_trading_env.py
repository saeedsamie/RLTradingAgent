import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest
from RL.trading_env import TradingEnv


class TestTradingEnv:
    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=1000, freq='5min')
        data = {
            'close': np.random.uniform(1200, 1300, 1000),
            'ma20': np.random.uniform(0.05, 0.06, 1000),
            'ma50': np.random.uniform(0.05, 0.06, 1000),
            'ma200': np.random.uniform(0.05, 0.06, 1000),
            'rsi': np.random.uniform(30, 70, 1000),
            'ichimoku_conversion': np.random.uniform(0.05, 0.06, 1000),
            'ichimoku_base': np.random.uniform(0.05, 0.06, 1000),
            'ichimoku_leading_a': np.random.uniform(0.05, 0.06, 1000),
            'ichimoku_leading_b': np.random.uniform(0.05, 0.06, 1000),
            'ichimoku_chikou': np.random.uniform(0.05, 0.06, 1000),
            'MACD_12_26_9': np.random.uniform(-0.01, 0.01, 1000),
            'MACDh_12_26_9': np.random.uniform(-0.01, 0.01, 1000),
            'MACDs_12_26_9': np.random.uniform(-0.01, 0.01, 1000),
            'STOCHk_14_3_3': np.random.uniform(20, 80, 1000),
            'STOCHd_14_3_3': np.random.uniform(20, 80, 1000),
            'atr': np.random.uniform(0.02, 0.04, 1000),
            'obv': np.random.uniform(0.1, 0.2, 1000)
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def env(self, sample_data):
        """Create a TradingEnv instance for testing."""
        return TradingEnv(sample_data, window_size=50, debug=False, max_episode_steps=100)

    def test_initialization(self, sample_data):
        """Test environment initialization."""
        env = TradingEnv(sample_data, window_size=50)
        
        assert env.window_size == 50
        assert env.current_step == 50
        assert env.position == 0
        assert env.balance == 1000.0
        assert env.equity == 1000.0
        assert len(env.feature_cols) == 16  # All columns except 'close'
        assert env.action_space.shape == (2,)
        assert env.observation_space.shape == (50, 19)  # window_size x (features + 3)

    def test_reset(self, env):
        """Test environment reset."""
        # Modify some state
        env.current_step = 100
        env.position = 1
        env.balance = 1500.0
        env.equity = 1600.0
        
        obs, info = env.reset()
        
        assert env.current_step == 50
        assert env.position == 0
        assert env.balance == 1000.0
        assert env.equity == 1000.0
        assert env.entry_price == 0
        assert env.lot_size == 0
        assert len(env.equity_curve) == 1
        assert env.equity_curve[0] == 1000.0
        assert obs.shape == (50, 19)
        assert isinstance(info, dict)

    def test_action_space_interpretation(self, env):
        """Test that different action formats are correctly interpreted."""
        env.reset()
        
        # Test numpy array
        action_np = np.array([1])  # Long
        obs, reward, done, truncated, info = env.step(action_np)
        assert env.position == 1
        assert env.lot_size == 0.01
        
        env.reset()
        
        # Test list
        action_list = [2]  # Short
        obs, reward, done, truncated, info = env.step(action_list)
        assert env.position == -1
        assert env.lot_size == 0.01

    def test_long_trade(self, env):
        """Test opening and closing a long position."""
        env.reset()
        price = env.df.iloc[50]['close']
        
        # Open long position
        obs, reward, done, truncated, info = env.step(1)
        assert env.position == 1
        assert env.entry_price == price
        assert env.lot_size == 0.01
        assert reward == 0  # No reward for opening position
        
        # Close long position (action 0 = out)
        obs, reward, done, truncated, info = env.step(0)
        assert env.position == 0
        assert env.entry_price == 0

    def test_short_trade(self, env):
        """Test opening and closing a short position."""
        env.reset()
        price = env.df.iloc[50]['close']
        
        # Open short position
        obs, reward, done, truncated, info = env.step(2)
        assert env.position == -1
        assert env.entry_price == price
        assert env.lot_size == 0.01
        
        # Close short position (action 0 = out)
        obs, reward, done, truncated, info = env.step(0)
        assert env.position == 0

    def test_episode_termination(self, env):
        """Test that episodes terminate correctly."""
        env.reset()
        
        # Run until episode should terminate
        step_count = 0
        while step_count < 150:  # More than max_episode_steps
            obs, reward, done, truncated, info = env.step(0)
            step_count += 1
            if done:
                break
        
        assert done
        assert step_count <= 100  # Should terminate at max_episode_steps

    def test_commission_calculation(self, env):
        """Test that commission is correctly calculated and deducted."""
        env.reset()
        initial_balance = env.balance
        
        # Open position with fixed lot size
        obs, reward, done, truncated, info = env.step(1)
        expected_commission = 0.01 * 0.5
        assert env.balance == initial_balance - expected_commission

    def test_pnl_calculation(self, env):
        """Test profit/loss calculation."""
        env.reset()
        
        # Open long position
        entry_price = env.df.iloc[50]['close']
        obs, reward, done, truncated, info = env.step(1)  # Fixed lot size
        lot_size = env.lot_size
        
        # Move to next step with higher price
        next_price = env.df.iloc[51]['close']
        obs, reward, done, truncated, info = env.step(0)  # Close position
        
        # Calculate expected PnL
        price_diff = next_price - entry_price
        expected_pnl = price_diff * lot_size * 10000
        expected_commission = lot_size * 0.5 * 2  # Commission for open and close
        expected_reward = expected_pnl - expected_commission
        
        assert abs(reward - expected_reward) < 0.01

    def test_equity_calculation(self, env):
        """Test equity calculation with floating PnL."""
        env.reset()
        
        # Open long position
        obs, reward, done, truncated, info = env.step(1)
        entry_price = env.entry_price
        lot_size = env.lot_size
        
        # Check equity with current price
        current_price = env.df.iloc[env.current_step]['close']
        expected_floating_pnl = (current_price - entry_price) * lot_size * 10000
        expected_equity = env.balance + expected_floating_pnl
        
        assert abs(env.equity - expected_equity) < 0.01

    def test_observation_space(self, env):
        """Test that observations are correctly formatted."""
        obs, info = env.reset()
        
        assert obs.shape == (50, 18)  # window_size x (features + 2)
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()

    def test_edge_cases(self, env):
        """Test edge cases and error handling."""
        env.reset()
        
        # Test invalid action (should be handled gracefully)
        obs, reward, done, truncated, info = env.step(5)  # Invalid action
        # Should not crash and should handle gracefully
        
        # Test negative action (should be handled gracefully)
        env.reset()
        obs, reward, done, truncated, info = env.step(-1)  # Invalid action
        # Should not crash and should handle gracefully

    def test_trade_statistics(self, env):
        """Test that trade statistics are correctly tracked."""
        env.reset()
        
        # Make some trades
        obs, reward, done, truncated, info = env.step(1)  # Open long
        obs, reward, done, truncated, info = env.step(0)  # Close long
        obs, reward, done, truncated, info = env.step(2)  # Open short
        obs, reward, done, truncated, info = env.step(0)  # Close short
        
        assert env.total_trades == 2
        assert env.total_commission > 0
        assert env.total_pnl != 0 