# !/usr/bin/env python3
"""
Checkpoint management utility for RL Trading Agent.
"""

import glob
import os
import re
import shutil


def list_checkpoints():
    """List all available checkpoints."""
    checkpoint_dir = 'models/checkpoints'
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints directory found.")
        return

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'ppo_trading_*_steps.zip'))

    if not checkpoint_files:
        print("No checkpoints found.")
        return

    print("Available checkpoints:")
    for file in sorted(checkpoint_files):
        # Extract timesteps from filename
        match = re.search(r'ppo_trading_(\d+)_steps\.zip', file)
        if match:
            timesteps = int(match.group(1))
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"  {os.path.basename(file)} - {timesteps:,} timesteps ({size_mb:.1f} MB)")
        else:
            print(f"  {os.path.basename(file)} - Unknown format")


def clear_checkpoints():
    """Remove all checkpoints."""
    checkpoint_dir = 'models/checkpoints'
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints directory found.")
        return

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'ppo_trading_*_steps.zip'))

    if not checkpoint_files:
        print("No checkpoints to remove.")
        return

    print(f"Found {len(checkpoint_files)} checkpoint(s) to remove:")
    for file in checkpoint_files:
        print(f"  {os.path.basename(file)}")

    confirm = input("Are you sure you want to remove all checkpoints? (y/N): ")
    if confirm.lower() == 'y':
        for file in checkpoint_files:
            os.remove(file)
            print(f"Removed: {os.path.basename(file)}")
        print("All checkpoints removed.")
    else:
        print("Operation cancelled.")


def get_latest_checkpoint():
    """Get the latest checkpoint file path."""
    checkpoint_dir = 'models/checkpoints'
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'ppo_trading_*_steps.zip'))
    latest_checkpoint = None
    latest_timesteps = 0

    for checkpoint_file in checkpoint_files:
        match = re.search(r'ppo_trading_(\d+)_steps\.zip', checkpoint_file)
        if match:
            timesteps = int(match.group(1))
            if timesteps > latest_timesteps:
                latest_timesteps = timesteps
                latest_checkpoint = checkpoint_file

    return latest_checkpoint, latest_timesteps


def backup_checkpoints():
    """Create a backup of all checkpoints."""
    checkpoint_dir = 'models/checkpoints'
    backup_dir = 'models/checkpoints_backup'

    if not os.path.exists(checkpoint_dir):
        print("No checkpoints directory found.")
        return

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'ppo_trading_*_steps.zip'))

    if not checkpoint_files:
        print("No checkpoints to backup.")
        return

    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)

    print(f"Backing up {len(checkpoint_files)} checkpoint(s)...")
    for file in checkpoint_files:
        filename = os.path.basename(file)
        backup_path = os.path.join(backup_dir, filename)
        shutil.copy2(file, backup_path)
        print(f"  Backed up: {filename}")

    print(f"Backup completed in: {backup_dir}")


def main():
    """Main function to handle command line arguments."""
    import sys

    if len(sys.argv) < 2:
        print("Checkpoint Management Utility")
        print("Usage:")
        print("  python scripts/manage_checkpoints.py list      - List all checkpoints")
        print("  python scripts/manage_checkpoints.py clear     - Remove all checkpoints")
        print("  python scripts/manage_checkpoints.py backup    - Backup all checkpoints")
        print("  python scripts/manage_checkpoints.py latest    - Show latest checkpoint")
        return

    command = sys.argv[1].lower()

    if command == 'list':
        list_checkpoints()
    elif command == 'clear':
        clear_checkpoints()
    elif command == 'backup':
        backup_checkpoints()
    elif command == 'latest':
        latest_checkpoint, latest_timesteps = get_latest_checkpoint()
        if latest_checkpoint:
            print(f"Latest checkpoint: {os.path.basename(latest_checkpoint)}")
            print(f"Timesteps: {latest_timesteps:,}")
        else:
            print("No checkpoints found.")
    else:
        print(f"Unknown command: {command}")
        print("Available commands: list, clear, backup, latest")


if __name__ == "__main__":
    main()
