"""
Script to calculate optimal total_timesteps for RL training.
"""

import pandas as pd

from scripts.config import get_config


def calculate_optimal_timesteps():
    """Calculate optimal total_timesteps based on data and configuration."""

    # Load data to get total size
    df = pd.read_csv('../data/processed/xauusd_5m_alpari_normalized.csv')
    total_data_points = len(df)

    # Get current configuration
    config = get_config('quarterly_focused')
    window_size = config['window_size']
    max_episode_steps = config['max_episode_steps']
    train_ratio = config['train_ratio']

    # Calculate available training data
    train_data_points = int(total_data_points * train_ratio)
    available_episodes = train_data_points // max_episode_steps

    print("=== TIMESTEPS CALCULATION ===")
    print(f"Total data points: {total_data_points:,}")
    print(f"Training data points: {train_data_points:,}")
    print(f"Window size: {window_size} bars ({window_size / 288:.1f} days)")
    print(f"Episode length: {max_episode_steps:,} bars ({max_episode_steps / 288:.1f} days)")
    print(f"Available episodes: {available_episodes:,}")

    # Calculate different timestep options
    print("\n=== TIMESTEPS OPTIONS ===")

    # Option 1: Experience all available episodes multiple times
    episodes_per_epoch = available_episodes
    timesteps_per_epoch = episodes_per_epoch * max_episode_steps

    print(f"1. Single epoch: {timesteps_per_epoch:,} timesteps")
    print(f"   - Episodes: {episodes_per_epoch:,}")
    print(f"   - Time: ~{timesteps_per_epoch / 1000000:.1f}M steps")

    # Option 2: Multiple epochs for better learning
    epochs = [5, 10, 20, 50, 100]
    for epoch in epochs:
        timesteps = timesteps_per_epoch * epoch
        print(f"{epoch}. {epoch} epochs: {timesteps:,} timesteps")
        print(f"   - Episodes: {episodes_per_epoch * epoch:,}")
        print(f"   - Time: ~{timesteps / 1000000:.1f}M steps")

    # Option 3: Based on data coverage
    print(f"\n3. Full data coverage: {train_data_points:,} timesteps")
    print(f"   - Episodes: {train_data_points // max_episode_steps:,}")
    print(f"   - Time: ~{train_data_points / 1000000:.1f}M steps")

    # Recommendation
    print("\n=== RECOMMENDATION ===")
    recommended_epochs = 20  # Good balance between learning and time
    recommended_timesteps = timesteps_per_epoch * recommended_epochs

    print(f"Recommended: {recommended_timesteps:,} timesteps ({recommended_epochs} epochs)")
    print(f"Reasoning:")
    print(f"- Allows agent to see all available episodes {recommended_epochs} times")
    print(f"- Sufficient for learning complex trading patterns")
    print(f"- Reasonable training time")
    print(f"- Good balance between exploration and exploitation")

    return recommended_timesteps


def calculate_training_time(timesteps):
    """Estimate training time based on timesteps."""
    # Rough estimate: 1000 timesteps per second (adjust based on your hardware)
    seconds = timesteps / 1000
    hours = seconds / 3600

    print(f"\n=== TRAINING TIME ESTIMATE ===")
    print(f"Timesteps: {timesteps:,}")
    print(f"Estimated time: {hours:.1f} hours ({seconds / 60:.1f} minutes)")
    print(f"Note: Actual time depends on your hardware (CPU/GPU)")


if __name__ == "__main__":
    optimal_timesteps = calculate_optimal_timesteps()
    calculate_training_time(optimal_timesteps)
