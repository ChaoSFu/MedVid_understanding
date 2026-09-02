"""Test HF_TOKEN configuration for repository access.

Run this script to verify that HF_TOKEN is properly configured
and has the necessary permissions for the private repo.
"""

import os
import sys
from huggingface_hub import HfApi
from pathlib import Path

REPO_ID = "UII-AI/MedVidBench-GroundTruth"
REPO_TYPE = "dataset"

def test_hf_token():
    """Test HF_TOKEN configuration."""
    print("=" * 80)
    print("TESTING HF_TOKEN CONFIGURATION")
    print("=" * 80)

    # Check if token exists
    print("\n[1/4] Checking HF_TOKEN environment variable...")
    token = os.environ.get('HF_TOKEN')

    if not token:
        print("❌ FAILED: HF_TOKEN not found in environment")
        print("\nHow to fix:")
        print("1. Generate token at: https://huggingface.co/settings/tokens")
        print("2. Grant 'write' permission to repositories")
        print("3. Set as environment variable:")
        print("   export HF_TOKEN='your_token_here'")
        print("\n4. For HuggingFace Spaces:")
        print("   Settings → Repository secrets → Add HF_TOKEN")
        sys.exit(1)

    print(f"✓ HF_TOKEN found (length: {len(token)} chars)")
    masked_token = token[:7] + "..." + token[-4:] if len(token) > 11 else "***"
    print(f"  Token: {masked_token}")

    # Initialize API
    print("\n[2/4] Initializing HuggingFace API...")
    try:
        api = HfApi()
        print("✓ HfApi initialized")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        sys.exit(1)

    # Test repository access (read)
    print(f"\n[3/4] Testing READ access to {REPO_ID}...")
    try:
        repo_info = api.repo_info(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token
        )
        print(f"✓ Successfully accessed repository")
        print(f"  Repo: {repo_info.id}")
        print(f"  Private: {repo_info.private}")
        print(f"  Last modified: {repo_info.lastModified}")

        # List files
        files = api.list_repo_files(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token
        )
        print(f"  Files in repo: {len(files)}")
        for file in files:
            print(f"    - {file}")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ FAILED: {error_msg}")

        if "401" in error_msg or "Unauthorized" in error_msg:
            print("\n→ Issue: Invalid or expired token")
            print("→ Fix: Regenerate token at https://huggingface.co/settings/tokens")
        elif "404" in error_msg or "Not Found" in error_msg:
            print(f"\n→ Issue: Repository '{REPO_ID}' not found")
            print("→ Fix: Create the repository:")
            print(f"  1. Go to: https://huggingface.co/new-dataset")
            print(f"  2. Owner: UII-AI")
            print(f"  3. Name: MedVidBench-GroundTruth")
            print(f"  4. Visibility: Private")
        elif "403" in error_msg or "Forbidden" in error_msg:
            print("\n→ Issue: No access to private repository")
            print("→ Fix: Ensure you're a member of UII-AI organization")

        sys.exit(1)

    # Test write access
    print(f"\n[4/4] Testing WRITE access to {REPO_ID}...")
    try:
        # Create a test file
        test_file = Path("test_upload.txt")
        with open(test_file, 'w') as f:
            f.write("Test upload to verify write permissions\n")

        print("  Creating test file...")
        result = api.upload_file(
            path_or_fileobj=str(test_file),
            path_in_repo="test_upload.txt",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
            commit_message="Test write access"
        )

        print(f"✓ Successfully uploaded test file")
        print(f"  Commit: {result}")

        # Clean up test file
        print("  Cleaning up test file...")
        api.delete_file(
            path_in_repo="test_upload.txt",
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            token=token,
            commit_message="Remove test file"
        )
        test_file.unlink()

        print(f"✓ Successfully deleted test file")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ FAILED: {error_msg}")

        if "403" in error_msg or "Forbidden" in error_msg:
            print("\n→ Issue: Token does not have write permission")
            print("→ Fix:")
            print("  1. Go to: https://huggingface.co/settings/tokens")
            print("  2. Create new token with WRITE permission")
            print("  3. Update HF_TOKEN environment variable")
        elif "401" in error_msg:
            print("\n→ Issue: Token invalid for write operations")
            print("→ Fix: Ensure token has 'write' scope")

        sys.exit(1)

    # Success
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print("\nYour HF_TOKEN is correctly configured with:")
    print("  ✓ Valid authentication")
    print("  ✓ Read access to private repository")
    print("  ✓ Write access to private repository")
    print("\nYou can now:")
    print("  1. Deploy app.py to HuggingFace Spaces")
    print("  2. Add HF_TOKEN as a Space secret")
    print("  3. Leaderboard will automatically sync to private repo")


if __name__ == "__main__":
    test_hf_token()
