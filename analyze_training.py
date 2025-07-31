import json

import matplotlib.pyplot as plt
import pandas as pd


def load_training_metrics(file_path='plots/training_metrics.json'):
    """Load training metrics from JSON file"""
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    return metrics


def analyze_metrics(metrics):
    """Analyze training metrics and print key insights"""
    print("=== TRAINING METRICS ANALYSIS ===\n")

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(metrics)

    print(f"Total episodes completed: {len(df)}")
    print(f"Total timesteps: {df['timesteps'].max():,}")
    print(f"Training time: {df['training_time_seconds'].max() / 60:.1f} minutes")
    print(f"Average episode length: {df['length'].mean():.0f} steps")

    print("\n=== REWARD ANALYSIS ===")
    print(f"Initial reward: {df['reward'].iloc[0]:.2f}")
    print(f"Final reward: {df['reward'].iloc[-1]:.2f}")
    print(f"Best reward: {df['reward'].max():.2f}")
    print(f"Worst reward: {df['reward'].min():.2f}")
    print(f"Reward improvement: {df['reward'].iloc[-1] - df['reward'].iloc[0]:.2f}")

    print("\n=== TRADING PERFORMANCE ===")
    print(f"Total trades across all episodes: {df['total_trades'].sum()}")
    print(f"Average trades per episode: {df['total_trades'].mean():.1f}")
    print(f"Best PnL: {df['total_pnl'].max():.2f}")
    print(f"Worst PnL: {df['total_pnl'].min():.2f}")
    print(f"Average PnL: {df['total_pnl'].mean():.2f}")

    print("\n=== LEARNING ANALYSIS ===")
    print(f"Initial loss: {df['loss'].iloc[0]:.6f}")
    print(f"Final loss: {df['loss'].iloc[-1]:.6f}")
    print(f"Loss reduction: {df['loss'].iloc[0] - df['loss'].iloc[-1]:.6f}")

    # Check for convergence
    recent_rewards = df['reward'].tail(3)
    if recent_rewards.std() < 1.0:
        print("✅ Agent appears to be converging (low reward variance)")
    else:
        print("⚠️ Agent still learning (high reward variance)")

    return df


def plot_metrics(df):
    """Create comprehensive plots of training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('RL Trading Agent Training Metrics', fontsize=16)

    # 1. Reward progression
    axes[0, 0].plot(df['timesteps'], df['reward'], 'b-', label='Episode Reward')
    axes[0, 0].plot(df['timesteps'], df['mean_reward_100'], 'r--', label='Moving Average')
    axes[0, 0].set_title('Reward Progression')
    axes[0, 0].set_xlabel('Timesteps')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Loss progression
    axes[0, 1].semilogy(df['timesteps'], df['loss'], 'g-')
    axes[0, 1].set_title('Loss Progression (Log Scale)')
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)

    # 3. Trading performance
    axes[0, 2].plot(df['timesteps'], df['total_pnl'], 'purple', label='Total PnL')
    axes[0, 2].plot(df['timesteps'], df['balance'], 'orange', label='Balance')
    axes[0, 2].set_title('Trading Performance')
    axes[0, 2].set_xlabel('Timesteps')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # 4. Number of trades
    axes[1, 0].plot(df['timesteps'], df['total_trades'], 'b-')
    axes[1, 0].set_title('Number of Trades per Episode')
    axes[1, 0].set_xlabel('Timesteps')
    axes[1, 0].set_ylabel('Trades')
    axes[1, 0].grid(True)

    # 5. Commission costs
    axes[1, 1].plot(df['timesteps'], df['total_commission'], 'r-')
    axes[1, 1].set_title('Commission Costs')
    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].set_ylabel('Commission')
    axes[1, 1].grid(True)

    # 6. Position status
    axes[1, 2].plot(df['timesteps'], df['position'], 'g-')
    axes[1, 2].set_title('Final Position (0=No Position, 1=Long)')
    axes[1, 2].set_xlabel('Timesteps')
    axes[1, 2].set_ylabel('Position')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig('plots/training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main analysis function"""
    try:
        metrics = load_training_metrics()
        df = analyze_metrics(metrics)
        plot_metrics(df)

        print("\n=== KEY INSIGHTS ===")

        # Analyze learning progression
        if df['reward'].iloc[-1] > df['reward'].iloc[0]:
            print("✅ Agent is improving (rewards increasing)")
        else:
            print("⚠️ Agent performance declining")

        # Analyze trading behavior
        recent_trades = df['total_trades'].tail(3).mean()
        if recent_trades < 5:
            print("⚠️ Agent has become very conservative (few trades)")
        elif recent_trades > 50:
            print("⚠️ Agent is overtrading")
        else:
            print("✅ Agent trading frequency looks reasonable")

        # Check for overfitting
        if df['loss'].iloc[-1] < 1e-10:
            print("⚠️ Very low loss - possible overfitting")

        print(f"\n📊 Analysis complete! Check 'plots/training_analysis.png' for visualizations.")

    except FileNotFoundError:
        print("❌ Training metrics file not found. Run training first.")
    except Exception as e:
        print(f"❌ Error analyzing metrics: {e}")


if __name__ == "__main__":
    main()
