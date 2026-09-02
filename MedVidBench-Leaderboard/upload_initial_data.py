"""Upload initial data to private HuggingFace repository.

This script uploads:
1. ground_truth.json - Private test set
2. leaderboard.json - Initial leaderboard with MedGRPO model

Run this once during initial setup.
"""

import os
import sys
import json
from pathlib import Path
from huggingface_hub import HfApi

# Configuration
REPO_ID = "UII-AI/MedVidBench-GroundTruth"
REPO_TYPE = "dataset"

def create_initial_leaderboard():
    """Create initial leaderboard.json with MedGRPO model data."""
    leaderboard_data = [
        {
            "rank": 1,
            "model_name": "uAI-NEXUS-MedVLM-1.0a-7B-RL",
            "organization": "UII",
            "cvs_acc": 0.914,
            "nap_acc": 0.427,
            "sa_acc": 0.244,
            "stg_miou": 0.202,
            "tag_miou_03": 0.216,
            "tag_miou_05": 0.156,
            "dvc_llm": 3.797,
            "dvc_f1": 0.210,
            "vs_llm": 4.184,
            "rc_llm": 3.442,
            "date": "2025-01-14",
            "contact": "gaozhongpai@gmail.com"
        }
    ]

    leaderboard_file = Path("leaderboard.json")
    with open(leaderboard_file, 'w') as f:
        json.dump(leaderboard_data, f, indent=2)

    print(f"✓ Created leaderboard.json with 1 entry (uAI-NEXUS-MedVLM-1.0a-7B-RL)")
    return leaderboard_file


def main():
    """Upload initial files to private repo."""
    # Check token
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("❌ HF_TOKEN environment variable not set")
        print("   Please run: export HF_TOKEN='your_token_here'")
        sys.exit(1)

    print("=" * 80)
    print(f"UPLOADING INITIAL DATA TO {REPO_ID}")
    print("=" * 80)

    api = HfApi()

    # 1. Upload ground truth
    print("\n[1/2] Uploading ground_truth.json...")
    ground_truth_file = Path("data/ground_truth.json")

    if not ground_truth_file.exists():
        print(f"   ❌ File not found: {ground_truth_file}")
        print(f"   Skipping ground_truth.json upload...")
    else:
        file_size = ground_truth_file.stat().st_size / (1024 * 1024)  # MB
        print(f"   File size: {file_size:.2f} MB")

        try:
            api.upload_file(
                path_or_fileobj=str(ground_truth_file),
                path_in_repo="ground_truth.json",
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                token=token,
                commit_message="Upload ground truth data"
            )
            print(f"   ✓ Uploaded ground_truth.json")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            sys.exit(1)

    # 2. Upload leaderboard
    print("\n[2/2] Uploading leaderboard.json...")

    # Create leaderboard with MedGRPO data
    leaderboard_file = create_initial_leaderboard()

    file_size = leaderboard_file.stat().st_size
    print(f"   File size: {file_size} bytes")

    try:
        api.upload_file(
            path_or_fileobj=str(leaderboard_file),
            path_in_repo="leaderboard.json",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
            commit_message="Initialize leaderboard with uAI-NEXUS-MedVLM-1.0a-7B-RL"
        )
        print(f"   ✓ Uploaded leaderboard.json")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✅ UPLOAD COMPLETE")
    print("=" * 80)
    print(f"\nRepository: https://huggingface.co/datasets/{REPO_ID}")
    print("\nUploaded:")
    print("  ✓ ground_truth.json (if available)")
    print("  ✓ leaderboard.json with uAI-NEXUS-MedVLM-1.0a-7B-RL")
    print("\nNext steps:")
    print("1. Verify files in repository")
    print("2. Add HF_TOKEN secret to HuggingFace Space")
    print("3. Deploy app.py to Space")
    print("4. Check app logs for successful loading")


if __name__ == "__main__":
    main()
