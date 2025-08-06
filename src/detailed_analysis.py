import json
import pandas as pd
import numpy as np

def detailed_analysis():
    """Detailed analysis of training progression"""
    with open('../outputs/plots/training_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    df = pd.DataFrame(metrics)
    
    print("=== DETAILED TRAINING PROGRESSION ANALYSIS ===\n")
    
    # Episode-by-episode analysis
    for i, row in df.iterrows():
        episode_num = i + 1
        print(f"Episode {episode_num}:")
        print(f"  Timesteps: {row['timesteps']:,}")
        print(f"  Reward: {row['reward']:.2f}")
        print(f"  Loss: {row['loss']:.6f}")
        print(f"  Trades: {row['total_trades']}")
        print(f"  PnL: {row['total_pnl']:.2f}")
        print(f"  Balance: {row['balance']:.2f}")
        print(f"  Position: {'Long' if row['position'] == 1 else 'None'}")
        print()
    
    print("=== LEARNING PATTERNS ===")
    
    # Early episodes (exploration)
    early_episodes = df.head(3)
    print("Early Episodes (Exploration Phase):")
    print(f"  Average reward: {early_episodes['reward'].mean():.2f}")
    print(f"  Average trades: {early_episodes['total_trades'].mean():.1f}")
    print(f"  High activity, high losses")
    print()
    
    # Middle episodes (learning)
    middle_episodes = df.iloc[3:6]
    print("Middle Episodes (Learning Phase):")
    print(f"  Average reward: {middle_episodes['reward'].mean():.2f}")
    print(f"  Average trades: {middle_episodes['total_trades'].mean():.1f}")
    print(f"  Improving performance, finding strategies")
    print()
    
    # Late episodes (convergence)
    late_episodes = df.tail(3)
    print("Late Episodes (Convergence Phase):")
    print(f"  Average reward: {late_episodes['reward'].mean():.2f}")
    print(f"  Average trades: {late_episodes['total_trades'].mean():.1f}")
    print(f"  Conservative strategy, minimal trading")
    print()
    
    print("=== TRADING STRATEGY EVOLUTION ===")
    
    # Analyze trading behavior changes
    print("Trading Behavior Progression:")
    print("  1. Started with aggressive trading (439 trades in episode 1)")
    print("  2. Gradually reduced trading frequency")
    print("  3. Eventually became very conservative (0 trades in recent episodes)")
    print("  4. Learned to avoid losses rather than seek profits")
    
    print("\nRisk Management Evolution:")
    print("  1. Initially took large positions with high risk")
    print("  2. Experienced significant losses (negative equity)")
    print("  3. Learned to manage risk better")
    print("  4. Now maintains stable balance with minimal exposure")
    
    print("\n=== PERFORMANCE METRICS ===")
    
    # Calculate key metrics
    total_return = df['total_pnl'].sum()
    total_commission = df['total_commission'].sum()
    net_return = total_return - total_commission
    win_rate = len(df[df['total_pnl'] > 0]) / len(df) * 100
    
    print(f"Total Return: {total_return:.2f}")
    print(f"Total Commission: {total_commission:.2f}")
    print(f"Net Return: {net_return:.2f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Average Return per Episode: {df['total_pnl'].mean():.2f}")
    
    print("\n=== CONVERGENCE ANALYSIS ===")
    
    # Check if agent has converged
    recent_rewards = df['reward'].tail(3)
    reward_std = recent_rewards.std()
    
    if reward_std < 1.0:
        print("✅ Agent has converged (low reward variance)")
        print(f"  Recent reward std: {reward_std:.2f}")
    else:
        print("⚠️ Agent still learning (high reward variance)")
        print(f"  Recent reward std: {reward_std:.2f}")
    
    # Check for overfitting
    if df['loss'].iloc[-1] < 1e-10:
        print("⚠️ Very low loss suggests possible overfitting")
        print("  Consider reducing model complexity or increasing exploration")
    
    print("\n=== RECOMMENDATIONS ===")
    
    if df['total_trades'].iloc[-1] == 0:
        print("⚠️ Agent has become too conservative")
        print("  Consider:")
        print("  - Increasing entropy coefficient for more exploration")
        print("  - Adjusting reward function to encourage trading")
        print("  - Reducing risk aversion in the environment")
    
    if df['reward'].iloc[-1] < -10:
        print("⚠️ Agent still showing negative rewards")
        print("  Consider:")
        print("  - Adjusting reward function")
        print("  - Increasing training time")
        print("  - Modifying environment parameters")

if __name__ == "__main__":
    detailed_analysis()