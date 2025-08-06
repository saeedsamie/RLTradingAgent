import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.RL import plot_equity_curve


def test_plot_equity_curve_creates_files(tmp_path):
    # Prepare sample data
    equity_curve = np.linspace(1000, 1200, 100)
    rewards = np.random.normal(0, 1, 100)
    returns = rewards
    drawdown = (equity_curve - np.maximum.accumulate(equity_curve)) / np.maximum.accumulate(equity_curve)
    window = 10

    # Change working directory to tmp_path for isolated test
    os.chdir(tmp_path)
    os.makedirs('plots', exist_ok=True)

    # Call plotting function
    plot_equity_curve(
        equity_curve,
        rewards=rewards,
        returns=returns,
        drawdown=drawdown,
        window=window
    )

    # List of expected plot files
    expected_files = [
        'equity_curve.png',
        'rewards_over_time.png',
        'drawdown_curve.png',
        'returns_histogram.png',
        'rolling_sharpe_ratio.png',
        'rolling_volatility.png',
        'cumulative_rewards.png',
        'returns_vs_equity_scatter.png',
        'equity_curve_with_ma.png',
    ]
    for fname in expected_files:
        fpath = os.path.join('plots', fname)
        assert os.path.exists(fpath), f"Plot file {fpath} was not created."
