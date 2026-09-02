#!/usr/bin/env python3
"""
Cleanup script to remove test/dummy submissions from leaderboard.

Usage:
    python cleanup_test_data.py --model-name "TestModel"
    python cleanup_test_data.py --all-test  # Remove all models with "test" in name
    python cleanup_test_data.py --list      # List all models
"""

import json
import argparse
import shutil
from pathlib import Path

# Configuration
PERSISTENT_DIR = Path("/data") if Path("/data").exists() else Path(".")
LEADERBOARD_FILE = PERSISTENT_DIR / "leaderboard.json"
RESULTS_DIR = PERSISTENT_DIR / "results"
SUBMISSIONS_DIR = PERSISTENT_DIR / "submissions"


def load_leaderboard():
    """Load leaderboard from file."""
    if not LEADERBOARD_FILE.exists():
        print(f"❌ Leaderboard file not found: {LEADERBOARD_FILE}")
        return []

    with open(LEADERBOARD_FILE, 'r') as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_leaderboard(data):
    """Save leaderboard to file."""
    # Update ranks
    for i, entry in enumerate(data, 1):
        entry['rank'] = i

    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved leaderboard with {len(data)} entries")


def list_models():
    """List all models in leaderboard."""
    data = load_leaderboard()

    if not data:
        print("Leaderboard is empty")
        return

    print(f"\n{'='*80}")
    print(f"LEADERBOARD MODELS ({len(data)} total)")
    print(f"{'='*80}\n")

    for entry in data:
        rank = entry.get('rank', '?')
        model_name = entry.get('model_name', 'Unknown')
        organization = entry.get('organization', 'Unknown')
        date = entry.get('date', 'Unknown')

        print(f"#{rank}: {model_name}")
        print(f"     Organization: {organization}")
        print(f"     Date: {date}")
        print()


def delete_model(model_name, dry_run=False):
    """Delete a model from leaderboard and cleanup associated files."""
    data = load_leaderboard()

    # Find the model
    model_entry = None
    for entry in data:
        if entry.get('model_name') == model_name:
            model_entry = entry
            break

    if not model_entry:
        print(f"❌ Model not found: {model_name}")
        return False

    print(f"\n{'='*80}")
    print(f"DELETING MODEL: {model_name}")
    print(f"{'='*80}\n")
    print(f"Organization: {model_entry.get('organization')}")
    print(f"Date: {model_entry.get('date')}")
    print(f"Rank: {model_entry.get('rank')}")

    if dry_run:
        print("\n⚠️  DRY RUN - No changes will be made")

    # Associated files/directories
    model_dir_name = model_name.replace(" ", "_")
    results_dir = RESULTS_DIR / model_dir_name

    # Remove from leaderboard
    if not dry_run:
        data = [e for e in data if e.get('model_name') != model_name]
        save_leaderboard(data)
        print(f"✓ Removed from leaderboard")
    else:
        print(f"[DRY RUN] Would remove from leaderboard")

    # Remove results directory
    if results_dir.exists():
        if not dry_run:
            shutil.rmtree(results_dir)
            print(f"✓ Removed results directory: {results_dir}")
        else:
            print(f"[DRY RUN] Would remove: {results_dir}")

    # Check for submission files (might not exist after evaluation)
    submission_dir = SUBMISSIONS_DIR / model_dir_name
    if submission_dir.exists():
        if not dry_run:
            shutil.rmtree(submission_dir)
            print(f"✓ Removed submissions directory: {submission_dir}")
        else:
            print(f"[DRY RUN] Would remove: {submission_dir}")

    print(f"\n{'='*80}")
    if dry_run:
        print("✓ DRY RUN COMPLETE - No changes made")
    else:
        print("✓ MODEL DELETED SUCCESSFULLY")
    print(f"{'='*80}\n")

    return True


def delete_test_models(dry_run=False):
    """Delete all models with 'test' in name (case-insensitive)."""
    data = load_leaderboard()

    test_models = [
        entry for entry in data
        if 'test' in entry.get('model_name', '').lower() or
           'test' in entry.get('organization', '').lower()
    ]

    if not test_models:
        print("No test models found")
        return

    print(f"\n{'='*80}")
    print(f"FOUND {len(test_models)} TEST MODELS")
    print(f"{'='*80}\n")

    for entry in test_models:
        print(f"- {entry.get('model_name')} ({entry.get('organization')})")

    print()

    if not dry_run:
        confirm = input("Delete all these models? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Cancelled")
            return

    for entry in test_models:
        model_name = entry.get('model_name')
        delete_model(model_name, dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Cleanup test/dummy submissions from MedVidBench leaderboard"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true",
                      help="List all models in leaderboard")
    group.add_argument("--model-name", type=str,
                      help="Delete specific model by name")
    group.add_argument("--all-test", action="store_true",
                      help="Delete all models with 'test' in name")

    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be deleted without actually deleting")

    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.model_name:
        delete_model(args.model_name, dry_run=args.dry_run)
    elif args.all_test:
        delete_test_models(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
