"""
Configuration file for RL Trading Agent with market cycle-based settings.
"""

# Market cycle configurations (in 5-minute bars)
MARKET_CYCLES = {
    'hourly': 12,           # 1 hour
    'daily': 288,           # 24 hours (24 * 12)
    'weekly': 2016,         # 7 days (7 * 288)
    'monthly': 8767,        # ~30.44 days
    'quarterly': 25920,     # ~90 days
    'yearly': 105192        # ~365.25 days
}

# Default configuration
DEFAULT_CONFIG = {
    'window_size': MARKET_CYCLES['daily'],      # Daily cycle for observation window
    'max_episode_steps': MARKET_CYCLES['weekly'],  # Weekly cycle for episode length
    'total_timesteps': 5_000_000,
    'debug': False,
    'commission_per_lot': 0.5,
    'lot_size': 0.01,
    'initial_balance': 1000.0
}

# Training configurations for different cycle lengths
TRAINING_CONFIGS = {
    'short_term': {
        'window_size': MARKET_CYCLES['hourly'],
        'max_episode_steps': MARKET_CYCLES['daily'],
        'total_timesteps': 2_000_000,
        'train_ratio': 0.8,
        'description': 'Hourly window, daily episodes - for short-term trading'
    },
    'medium_term': {
        'window_size': MARKET_CYCLES['daily'],
        'max_episode_steps': MARKET_CYCLES['weekly'], 
        'total_timesteps': 5_000_000,
        'train_ratio': 0.8,
        'description': 'Daily window, weekly episodes - balanced approach with better market cycles'
    },
    'long_term': {
        'window_size': MARKET_CYCLES['weekly'],
        'max_episode_steps': MARKET_CYCLES['monthly'],
        'total_timesteps': 10_000_000,
        'train_ratio': 0.8,
        'description': 'Weekly window, monthly episodes - for trend following'
    },
    'trend_following': {
        'window_size': MARKET_CYCLES['monthly'],
        'max_episode_steps': MARKET_CYCLES['quarterly'],
        'total_timesteps': 20_000_000,
        'train_ratio': 0.8,
        'description': 'Monthly window, quarterly episodes - for long-term trends'
    },
    'quarterly_focused': {
        'window_size': MARKET_CYCLES['daily'],
        'max_episode_steps': MARKET_CYCLES['quarterly'],
        'total_timesteps': 52_800_000,  # 60 epochs of quarterly episodes
        'train_ratio': 0.8,
        'description': 'Daily window, quarterly episodes - optimal for market cycle learning'
    },
    'active_trading': {
        'window_size': MARKET_CYCLES['daily'],
        'max_episode_steps': MARKET_CYCLES['weekly'],  # Shorter episodes for more frequent learning
        'total_timesteps': 5_000_000,  # Faster training
        'train_ratio': 0.8,
        'description': 'Daily window, weekly episodes - optimized for active trading with enhanced exploration'
    },
    'improved_trading': {
        'window_size': MARKET_CYCLES['daily'],
        'max_episode_steps': MARKET_CYCLES['weekly'],
        'total_timesteps': 5_000_000,
        'train_ratio': 0.8,
        'description': 'Improved configuration with better reward function and hyperparameters to encourage trading'
    },
    'ultra_aggressive': {
        'window_size': MARKET_CYCLES['daily'],
        'max_episode_steps': MARKET_CYCLES['weekly'],
        'total_timesteps': 3_000_000,  # Shorter training for faster iteration
        'train_ratio': 0.8,
        'description': 'Ultra aggressive configuration to force trading activity'
    },
    'deep_network': {
        'window_size': MARKET_CYCLES['daily'],
        'max_episode_steps': MARKET_CYCLES['weekly'],
        'total_timesteps': 5_000_000,  # Longer training for deep network
        'train_ratio': 0.8,
        'description': 'Deep network configuration for complex pattern learning'
    }
}

def get_config(config_name='quarterly_focused'):
    """Get configuration by name."""
    if config_name in TRAINING_CONFIGS:
        return TRAINING_CONFIGS[config_name]
    else:
        print(f"Warning: Config '{config_name}' not found. Using default.")
        return TRAINING_CONFIGS['quarterly_focused']

def print_available_configs():
    """Print all available configurations."""
    print("Available configurations:")
    for name, config in TRAINING_CONFIGS.items():
        print(f"  {name}: {config['description']}")
        print(f"    Window: {config['window_size']} bars ({config['window_size']/288:.1f} days)")
        print(f"    Episode: {config['max_episode_steps']} bars ({config['max_episode_steps']/288:.1f} days)")
        print(f"    Timesteps: {config['total_timesteps']:,} ({config['total_timesteps']/1000000:.1f}M)")
        print()

if __name__ == "__main__":
    print_available_configs()
    print("Testing configuration retrieval:")
    config = get_config('medium_term')
    print(f"Medium term config: {config}")