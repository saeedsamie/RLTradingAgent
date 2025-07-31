#!/usr/bin/env python3
"""
Script to restart training with enhanced parameters for more active trading.
This applies the parameter adjustments we made to encourage more trading activity.
"""

import glob
import os
import shutil
from datetime import datetime


def backup_current_training():
    """Backup current training state"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/training_backup_{timestamp}"

    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)

    # Backup current model and checkpoints
    if os.path.exists("models/ppo_trading.zip"):
        shutil.copy("models/ppo_trading.zip", f"{backup_dir}/")

    # Backup checkpoints
    checkpoint_files = glob.glob("models/checkpoints/ppo_trading_*_steps.zip")
    for file in checkpoint_files:
        shutil.copy(file, f"{backup_dir}/")

    # Backup training metrics
    if os.path.exists("plots/training_metrics.json"):
        shutil.copy("plots/training_metrics.json", f"{backup_dir}/")

    print(f"✅ Current training state backed up to: {backup_dir}")
    return backup_dir


def clear_training_state():
    """Clear current training state for fresh start"""
    # Remove current model
    if os.path.exists("models/ppo_trading.zip"):
        os.remove("models/ppo_trading.zip")
        print("🗑️ Removed current model")

    # Remove checkpoints
    checkpoint_files = glob.glob("models/checkpoints/ppo_trading_*_steps.zip")
    for file in checkpoint_files:
        os.remove(file)
        print(f"🗑️ Removed checkpoint: {file}")

    # Clear training metrics
    if os.path.exists("plots/training_metrics.json"):
        os.remove("plots/training_metrics.json")
        print("🗑️ Cleared training metrics")

    print("✅ Training state cleared for fresh start")


def modify_config_for_active_trading():
    """Modify the run script to use active trading config"""
    # Read current run script
    with open("scripts/run_rl_pipeline.py", "r") as f:
        content = f.read()

    # Replace the config line
    old_config = "config = get_config('quarterly_focused')"
    new_config = "config = get_config('active_trading')"

    if old_config in content:
        content = content.replace(old_config, new_config)

        # Write back the modified content
        with open("scripts/run_rl_pipeline.py", "w") as f:
            f.write(content)

        print("✅ Modified configuration to use 'active_trading'")
    else:
        print("⚠️ Could not find config line to modify")


def main():
    """Main function to restart training with active trading parameters"""
    print("🚀 Restarting Training with Active Trading Parameters")
    print("=" * 60)

    # 1. Backup current training
    backup_dir = backup_current_training()

    # 2. Clear training state
    clear_training_state()

    # 3. Modify configuration
    modify_config_for_active_trading()

    print("\n" + "=" * 60)
    print("✅ Ready to restart training with enhanced parameters!")
    print("\n📋 Parameter Changes Applied:")
    print("  • Higher entropy coefficient (0.01 → 0.05)")
    print("  • Higher learning rate (3e-4 → 5e-4)")
    print("  • Enhanced diversity bonus in reward function")
    print("  • Trading activity tracking and bonuses")
    print("  • Shorter episodes (quarterly → weekly)")
    print("  • Faster training (17.6M → 5M timesteps)")

    print("\n🎯 Expected Improvements:")
    print("  • More trading activity")
    print("  • Better exploration")
    print("  • Faster learning")
    print("  • Reduced over-conservatism")

    print(f"\n💾 Backup saved to: {backup_dir}")
    print("\n🚀 Run the following command to start training:")
    print("   python -m scripts.run_rl_pipeline --fresh-start")


if __name__ == "__main__":
    main()
