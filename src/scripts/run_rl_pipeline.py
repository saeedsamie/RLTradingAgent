import json
import os
import sys
from pathlib import Path
import glob
import shutil

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.RL.evaluate import analyze_trades as evaluate_with_trade_analysis
from src.RL.plotting import plot_equity_curve
from src.RL.train_agent import train_agent
from src.config.config import get_config
from src.scripts.data_prep import load_data, check_missing_intervals

# Configuration - using normalized improved dataset for better reward improvement
DATA_PATH = 'data/processed/xauusd_5m_improved_normalized.csv'
MODEL_PATH = 'outputs/models/ppo_trading.zip'

CONFIG = get_config('deep_network')
WINDOW_SIZE = CONFIG['window_size']
MAX_EPISODE_STEPS = CONFIG['max_episode_steps']
TOTAL_TIMESTEPS = CONFIG['total_timesteps']
TRAIN_RATIO = CONFIG['train_ratio']

# Add command line argument for fresh start
import sys


def handle_fresh_start():

    print(f"\n=== Fresh start requested ===")
    checkpoint_files = glob.glob('outputs/models/checkpoints/ppo_trading_*_steps.zip')
    for file in checkpoint_files:
        os.remove(file)
        print(f"Removed: {file}")
    print("Starting training from scratch.")

    # Clean up the 'plots' and 'trade_history' directories for a fresh start
    for dir_path in ['outputs/plots', 'outputs/trade_history']:
        if os.path.exists(dir_path):
            print(f"Clearing directory: {dir_path}")
            shutil.rmtree(dir_path)
            print(f"Removed: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created empty directory: {dir_path}")


FRESH_START = '--fresh-start' in sys.argv
DEBUG_MODE = '--debug' in sys.argv
CPU_ONLY = '--cpu-only' in sys.argv

if __name__ == '__main__':
    # Show help if requested
    if '--help' in sys.argv or '-h' in sys.argv:
        print("RL Trading Agent Pipeline")
        print("=" * 30)
        print("Usage: python scripts/run_rl_pipeline.py [options]")
        print("\nOptions:")
        print("  --fresh-start    Clear all checkpoints and start training from scratch")
        print("  --debug          Enable debug logging")
        print("  --cpu-only       Force CPU usage (skip GPU acceleration)")
        print("  --help, -h       Show this help message")
        print("\nExamples:")
        print("  python scripts/run_rl_pipeline.py                    # Normal training")
        print("  python scripts/run_rl_pipeline.py --cpu-only         # CPU-only training")
        print("  python scripts/run_rl_pipeline.py --fresh-start      # Fresh start")
        print("  python scripts/run_rl_pipeline.py --debug --cpu-only # Debug with CPU")
        sys.exit(0)

    print(f"Using configuration: {CONFIG['description']}")
    print(f"Window size: {WINDOW_SIZE} bars ({WINDOW_SIZE / 288:.1f} days)")
    print(f"Episode length: {MAX_EPISODE_STEPS} bars ({MAX_EPISODE_STEPS / 288:.1f} days)")
    print(f"Total timesteps: {TOTAL_TIMESTEPS:,} ({TOTAL_TIMESTEPS / 1000000:.1f}M)")
    if CPU_ONLY:
        print("⚠️  CPU-only mode enabled")
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

    # Train agent
    model = train_agent(
        train_df,
        model_path=MODEL_PATH,
        window_size=WINDOW_SIZE,
        max_episode_steps=MAX_EPISODE_STEPS,
        total_timesteps=TOTAL_TIMESTEPS,
        debug=DEBUG_MODE,
        force_cpu=CPU_ONLY
    )

    # Evaluate agent with comprehensive trade analysis
    print(f"\n=== Evaluation Debug Info ===")
    print(f"Model path: {MODEL_PATH}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Test data columns: {list(test_df.columns)}")
    print(f"Test data date range: {test_df.iloc[0][test_df.columns[0]]} to {test_df.iloc[-1][test_df.columns[0]]}")

    results = evaluate_with_trade_analysis(
        model_path=MODEL_PATH,
        test_df=test_df,
        window_size=WINDOW_SIZE,
        max_episode_steps=MAX_EPISODE_STEPS
    )

    # Extract results
    rewards = results['rewards']
    equity_curve = results['equity_curve']
    trades = results['trades']
    trade_stats = results['trade_stats']

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

    # Write comprehensive trade data to CSV with datetime information
    import csv

    trades_csv_path = "outputs/plots/trades.csv"
    with open(trades_csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "entry_idx", "exit_idx", "entry_datetime", "exit_datetime",
            "entry_price", "exit_price", "entry_equity", "exit_equity",
            "position_type", "pnl", "commission", "net_pnl",
            "duration", "return_pct", "win"
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
                trade['duration'], trade['return_pct'], trade['win']
            ])
    print(f"Comprehensive trade data with datetime written to {trades_csv_path}")

    # Write trade statistics to CSV
    stats_csv_path = "outout/plots/trade_statistics.csv"
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
    results_json_path = "outputs/plots/evaluation_results.json"

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
    os.makedirs('outputs/plots', exist_ok=True)

    with open(results_json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Comprehensive results saved to {results_json_path}")

    print(f"\n=== Files Generated ===")
    print(f"1. {trades_csv_path} - Detailed trade data")
    print(f"2. {stats_csv_path} - Trade statistics")
    print(f"3. {results_json_path} - Complete evaluation results")
    print(f"4. outputs/plots/equity_curve.png - Equity curve visualization")
