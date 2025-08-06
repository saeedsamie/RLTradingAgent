import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

from src.RL.trading_env import TradingEnv


def evaluate_agent(model_path, test_df, window_size=288, max_episode_steps=25920):
    """Run the trained agent on test data and collect results. Uses MlpPolicy."""
    # Select only the allowed feature columns and 'close' for price
    feature_cols = [
        'ma20', 'ma50', 'ma200', 'rsi', 'ichimoku_conversion', 'ichimoku_base',
        'ichimoku_leading_a', 'ichimoku_leading_b', 'ichimoku_chikou',
        'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'STOCHk_14_3_3', 'STOCHd_14_3_3', 'atr', 'obv'
    ]
    # Keep 'close' for price calculation in env and preserve datetime index
    filtered_df = test_df[['close'] + feature_cols].copy()
    # Ensure datetime index is preserved - check both index and first column
    if hasattr(test_df.index, 'dtype') and test_df.index.dtype == 'datetime64[ns]':
        filtered_df.index = test_df.index
    elif len(test_df.columns) > 0:
        # Check if first column is datetime
        first_col = test_df.columns[0]
        if 'time' in first_col.lower() or 'date' in first_col.lower():
            # Use first column as datetime index
            filtered_df.index = pd.to_datetime(test_df[first_col])
            # Remove the datetime column from features
            if first_col in filtered_df.columns:
                filtered_df = filtered_df.drop(columns=[first_col])
    env = TradingEnv(filtered_df, window_size=window_size, debug=False, max_episode_steps=max_episode_steps)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check if model file exists
    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device=device)
    print(f"Model loaded successfully on device: {device}")
    obs, _ = env.reset()
    done = False
    rewards = []
    while not done:
        # Model outputs a discrete action (0, 1, or 2)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
    equity_curve = env.equity_curve
    return rewards, equity_curve, env


def sharpe_ratio(returns, risk_free_rate=0):
    returns = np.array(returns)
    if returns.std() == 0:
        return 0
    return (returns.mean() - risk_free_rate) / (returns.std() + 1e-8) * np.sqrt(252 * 24 * 12)


def max_drawdown(equity_curve):
    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return drawdown.min()
    # Get the trades and analyze them

def get_datetime_at_index(env, index):
    """
    Get datetime for a specific index using the environment's datetime function.
    
    Args:
        env: Trading environment with datetime functionality
        index (int): The index to get datetime for
        
    Returns:
        datetime or None: The datetime at the given index, or None if not available
    """
    if env is not None and hasattr(env, 'get_datetime_at_index'):
        return env.get_datetime_at_index(index)
    return None

def extract_trades(equity_curve, env=None):
    """
    Extract actual trades from the trading environment's trade history.
    Returns a list of trade dictionaries with comprehensive trade information.
    
    Args:
        equity_curve: List of equity values (for backward compatibility)
        env: Trading environment with trade history
    """
    trades = []
    
    # Use actual trade history from environment if available
    if env is not None and hasattr(env, 'trade_history') and env.trade_history:
        for trade in env.trade_history:
            entry_idx = trade['entry_idx']
            exit_idx = trade['exit_idx']
            entry_price = trade['entry_price']
            exit_price = trade['exit_price']
            position_type = trade['position_type']
            pnl = trade['pnl']
            commission = trade['commission']
            duration = trade['duration']
            
            # Get datetime information using the centralized function
            entry_datetime = get_datetime_at_index(env, entry_idx)
            exit_datetime = get_datetime_at_index(env, exit_idx)
            
            # Calculate equity values
            entry_equity = equity_curve[entry_idx] if entry_idx is not None and entry_idx < len(equity_curve) else 0
            exit_equity = equity_curve[exit_idx] if exit_idx < len(equity_curve) else 0
            
            # Calculate return percentage
            return_pct = (pnl / entry_equity * 100) if entry_equity != 0 else 0
            
            trade_info = {
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_datetime': entry_datetime,
                'exit_datetime': exit_datetime,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_equity': entry_equity,
                'exit_equity': exit_equity,
                'position_type': position_type,
                'pnl': pnl,
                'commission': commission,
                'net_pnl': pnl - commission,
                'duration': duration,
                'return_pct': return_pct,
                'win': pnl > 0
            }
            
            trades.append(trade_info)
    
    # Fallback to equity curve analysis if no trade history available
    if not trades:
        # Original equity curve analysis (less accurate but provides fallback)
        in_trade = False
        entry_idx = None
        entry_equity = None
        entry_price = None
        position_type = None

        for i in range(1, len(equity_curve)):
            if not in_trade:
                if equity_curve[i] > equity_curve[i-1]:
                    in_trade = True
                    entry_idx = i-1
                    entry_equity = equity_curve[i-1]
                    if env is not None and hasattr(env, 'df') and entry_idx < len(env.df):
                        entry_price = env.df.iloc[entry_idx]['close']
                    else:
                        entry_price = None
                    if i < len(equity_curve) - 1:
                        if equity_curve[i+1] > equity_curve[i]:
                            position_type = 'long'
                        else:
                            position_type = 'short'
                    else:
                        position_type = 'unknown'
            else:
                if equity_curve[i] <= equity_curve[i-1]:
                    exit_idx = i-1
                    exit_equity = equity_curve[exit_idx]
                    pnl = exit_equity - entry_equity
                    
                    if env is not None and hasattr(env, 'df') and exit_idx < len(env.df):
                        exit_price = env.df.iloc[exit_idx]['close']
                    else:
                        exit_price = None
                    
                    commission = 0
                    if env is not None and hasattr(env, 'lot_size') and hasattr(env, 'commission_per_lot'):
                        commission = env.lot_size * env.commission_per_lot
                    
                    duration = exit_idx - entry_idx
                    return_pct = (pnl / entry_equity * 100) if entry_equity != 0 else 0
                    
                    # Get datetime information using the centralized function
                    entry_datetime = get_datetime_at_index(env, entry_idx)
                    exit_datetime = get_datetime_at_index(env, exit_idx)
                    
                    trade_info = {
                        'entry_idx': entry_idx,
                        'exit_idx': exit_idx,
                        'entry_datetime': entry_datetime,
                        'exit_datetime': exit_datetime,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'entry_equity': entry_equity,
                        'exit_equity': exit_equity,
                        'position_type': position_type,
                        'pnl': pnl,
                        'commission': commission,
                        'net_pnl': pnl - commission,
                        'duration': duration,
                        'return_pct': return_pct,
                        'win': pnl > 0
                    }
                    
                    trades.append(trade_info)
                    in_trade = False
        
        if in_trade:
            exit_idx = len(equity_curve) - 1
            exit_equity = equity_curve[exit_idx]
            pnl = exit_equity - entry_equity
            
            if env is not None and hasattr(env, 'df') and exit_idx < len(env.df):
                exit_price = env.df.iloc[exit_idx]['close']
            else:
                exit_price = None
            
            commission = 0
            if env is not None and hasattr(env, 'lot_size') and hasattr(env, 'commission_per_lot'):
                commission = env.lot_size * env.commission_per_lot
            
            duration = exit_idx - entry_idx
            return_pct = (pnl / entry_equity * 100) if entry_equity != 0 else 0
            
            # Get datetime information using the centralized function
            entry_datetime = get_datetime_at_index(env, entry_idx)
            exit_datetime = get_datetime_at_index(env, exit_idx)
            
            trade_info = {
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'entry_datetime': entry_datetime,
                'exit_datetime': exit_datetime,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_equity': entry_equity,
                'exit_equity': exit_equity,
                'position_type': position_type,
                'pnl': pnl,
                'commission': commission,
                'net_pnl': pnl - commission,
                'duration': duration,
                'return_pct': return_pct,
                'win': pnl > 0
            }
            
            trades.append(trade_info)
    
    return trades

def analyze_trades(trades):
    """
    Given a list of trade dictionaries, compute statistics.
    Returns a dict with number of trades, win rate, average win, average loss, expectancy, etc.
    """
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "expectancy": 0,
            "max_win": 0,
            "max_loss": 0,
            "avg_trade_duration": 0,
            "avg_commission": 0,
            "total_commission": 0,
            "avg_return_pct": 0,
            "long_trades": 0,
            "short_trades": 0,
            "long_win_rate": 0,
            "short_win_rate": 0
        }
    
    # Extract data from trade dictionaries
    pnls = [t['pnl'] for t in trades]
    net_pnls = [t['net_pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    durations = [t['duration'] for t in trades]
    commissions = [t['commission'] for t in trades]
    return_pcts = [t['return_pct'] for t in trades]
    
    # Position type analysis
    long_trades = [t for t in trades if t['position_type'] == 'long']
    short_trades = [t for t in trades if t['position_type'] == 'short']
    
    long_wins = [t for t in long_trades if t['win']]
    short_wins = [t for t in short_trades if t['win']]
    
    num_trades = len(trades)
    win_rate = len(wins) / num_trades if num_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    max_win = np.max(wins) if wins else 0
    max_loss = np.min(losses) if losses else 0
    avg_trade_duration = np.mean(durations) if durations else 0
    avg_commission = np.mean(commissions) if commissions else 0
    total_commission = np.sum(commissions) if commissions else 0
    avg_return_pct = np.mean(return_pcts) if return_pcts else 0
    
    long_win_rate = len(long_wins) / len(long_trades) if long_trades else 0
    short_win_rate = len(short_wins) / len(short_trades) if short_trades else 0
    
    return {
        "num_trades": num_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "max_win": max_win,
        "max_loss": max_loss,
        "avg_trade_duration": avg_trade_duration,
        "avg_commission": avg_commission,
        "total_commission": total_commission,
        "avg_return_pct": avg_return_pct,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate
    }


def evaluate_with_trade_analysis(model_path, test_df, window_size=288, max_episode_steps=25920):
    """
    Enhanced evaluation function that includes comprehensive trade analysis.
    Returns rewards, equity curve, and detailed trade statistics.
    """
    # Run the agent evaluation
    rewards, equity_curve, env = evaluate_agent(model_path, test_df, window_size, max_episode_steps)
    
    # Extract trades with full information including datetime
    trades = extract_trades(equity_curve, env)
    
    # Debug trade extraction
    print(f"Extracted {len(trades)} trades")
    if env is not None and hasattr(env, 'trade_history'):
        print(f"Environment trade history: {len(env.trade_history)} trades")
        if env.trade_history:
            print(f"First trade: {env.trade_history[0]}")
            print(f"Last trade: {env.trade_history[-1]}")
    
    # Analyze trades
    trade_stats = analyze_trades(trades)
    
    # Calculate additional metrics
    sharpe = sharpe_ratio(rewards)
    max_dd = max_drawdown(equity_curve)
    
    # Combine all results
    results = {
        'rewards': rewards,
        'equity_curve': equity_curve,
        'trades': trades,
        'trade_stats': trade_stats,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0] if equity_curve else 0
    }
    
    return results
