"""
MedVidBench Leaderboard - Interactive leaderboard for evaluating Video-Language Models
on the MedVidBench benchmark across 8 medical video understanding tasks.
"""

import gradio as gr
import pandas as pd
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from huggingface_hub import hf_hub_download, HfApi

def load_ground_truth():
    """
    Load ground truth from private HuggingFace dataset repository.
    Falls back to local file for development.
    """
    try:
        # Get token from environment (HF Space secret)
        token = os.environ.get('HF_TOKEN')

        if not token:
            print("⚠️  HF_TOKEN not found in environment, trying local file...")
            raise ValueError("HF_TOKEN not found")

        # Download from private repository
        print("⏳ Downloading ground truth from private repository...")
        gt_file = hf_hub_download(
            repo_id="UII-AI/MedVidBench-GroundTruth",
            filename="ground_truth.json",
            repo_type="dataset",
            token=token,
            cache_dir="./cache"  # Cache locally to avoid re-downloading
        )

        # Load the data
        with open(gt_file) as f:
            data = json.load(f)

        print(f"✓ Loaded ground truth from private repo: {len(data)} samples")
        return data

    except Exception as e:
        print(f"⚠️  Could not load from private repo: {e}")

        # Fallback to local file for development
        local_file = Path("data/ground_truth.json")
        if local_file.exists():
            with open(local_file) as f:
                data = json.load(f)
            print(f"✓ Loaded ground truth from local file: {len(data)} samples")
            return data
        else:
            raise FileNotFoundError(
                "Ground truth not found. Please set HF_TOKEN secret or provide local file."
            )

# Configuration - Use persistent storage on HuggingFace Spaces
# On HF Spaces, /data is persistent across app updates
PERSISTENT_DIR = Path("/data") if Path("/data").exists() else Path(".")

SUBMISSIONS_DIR = PERSISTENT_DIR / "submissions"
RESULTS_DIR = PERSISTENT_DIR / "results"
LEADERBOARD_FILE = PERSISTENT_DIR / "leaderboard.json"
OFFICIAL_LEADERBOARD_FILE = PERSISTENT_DIR / "official_leaderboard.json"
EVAL_SCRIPT = Path("evaluation/evaluate_all_pai.py")  # Local copy in repo

# Ensure directories exist
SUBMISSIONS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Load ground truth at startup
print("=" * 60)
print("LOADING GROUND TRUTH DATA")
print("=" * 60)
GROUND_TRUTH = load_ground_truth()
print("=" * 60)

# Ensure ground truth is available at expected path for evaluation subprocess
GROUND_TRUTH_FILE = Path("data/ground_truth.json")
if not GROUND_TRUTH_FILE.exists():
    print(f"⚠️  Saving ground truth to {GROUND_TRUTH_FILE} for evaluation subprocess...")
    GROUND_TRUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(GROUND_TRUTH, f)
    print(f"✓ Ground truth saved to {GROUND_TRUTH_FILE}")

# MedVidBench Metrics Definitions (10 metrics from 8 tasks)
# Note: TAL has 2 metrics, DVC has 2 metrics, others have 1 metric each
METRICS = {
    "cvs_acc": {
        "name": "CVS_acc",
        "full_name": "CVS Assessment Accuracy",
        "higher_better": True,
        "description": "Clinical variable scoring accuracy"
    },
    "nap_acc": {
        "name": "NAP_acc",
        "full_name": "Next Action Prediction Accuracy",
        "higher_better": True,
        "description": "Accuracy in predicting next surgical step"
    },
    "sa_acc": {
        "name": "SA_acc",
        "full_name": "Skill Assessment Accuracy",
        "higher_better": True,
        "description": "Surgical skill level evaluation accuracy"
    },
    "stg_miou": {
        "name": "STG_mIoU",
        "full_name": "Spatiotemporal Grounding mIoU",
        "higher_better": True,
        "description": "Mean IoU for spatial+temporal localization"
    },
    "tag_miou_03": {
        "name": "TAG_mIoU@0.3",
        "full_name": "Temporal Action Grounding mIoU@0.3",
        "higher_better": True,
        "description": "Mean IoU at threshold 0.3 for temporal localization"
    },
    "tag_miou_05": {
        "name": "TAG_mIoU@0.5",
        "full_name": "Temporal Action Grounding mIoU@0.5",
        "higher_better": True,
        "description": "Mean IoU at threshold 0.5 for temporal localization"
    },
    "dvc_f1": {
        "name": "DVC_F1",
        "full_name": "Dense Video Captioning F1",
        "higher_better": True,
        "description": "F1 score for temporal segment localization"
    },
    "dvc_llm": {
        "name": "DVC_llm",
        "full_name": "Dense Video Captioning LLM Score",
        "higher_better": True,
        "description": "Caption quality score (LLM judge or semantic similarity)"
    },
    "vs_llm": {
        "name": "VS_llm",
        "full_name": "Video Summary LLM Score",
        "higher_better": True,
        "description": "Video summary quality score"
    },
    "rc_llm": {
        "name": "RC_llm",
        "full_name": "Region Caption LLM Score",
        "higher_better": True,
        "description": "Region caption quality score"
    },
}

# Task descriptions for Tasks & Metrics tab
TASKS = {
    "tal": {
        "name": "Temporal Action Localization (TAL)",
        "key": "tal",
        "metrics": "TAG_mIoU@0.3, TAG_mIoU@0.5",
        "description": "Identify and temporally localize surgical actions in video"
    },
    "stg": {
        "name": "Spatiotemporal Grounding (STG)",
        "key": "stg",
        "metrics": "STG_mIoU",
        "description": "Localize objects in both space (bbox) and time (temporal span)"
    },
    "next_action": {
        "name": "Next Action Prediction (NAP)",
        "key": "next_action",
        "metrics": "NAP_acc",
        "description": "Predict the next surgical step given current video context"
    },
    "dvc": {
        "name": "Dense Video Captioning (DVC)",
        "key": "dvc",
        "metrics": "DVC_llm, DVC_F1",
        "description": "Generate captions for multiple events with temporal localization"
    },
    "vs": {
        "name": "Video Summary (VS)",
        "key": "vs",
        "metrics": "VS_llm",
        "description": "Generate comprehensive summary of surgical procedure"
    },
    "rc": {
        "name": "Region Caption (RC)",
        "key": "rc",
        "metrics": "RC_llm",
        "description": "Describe specific spatial regions in surgical frames"
    },
    "skill_assessment": {
        "name": "Skill Assessment (SA)",
        "key": "skill_assessment",
        "metrics": "SA_acc",
        "description": "Evaluate surgeon skill level (novice/intermediate/expert)"
    },
    "cvs_assessment": {
        "name": "CVS Assessment",
        "key": "cvs_assessment",
        "metrics": "CVS_acc",
        "description": "Score clinical variables in surgical performance"
    },
}

# Test set statistics
TEST_SET_STATS = {
    "total_samples": 6245,
    "datasets": ["AVOS", "CholecT50", "CholecTrack20", "Cholec80_CVS", "CoPESD", "EgoSurgery", "NurViD", "JIGSAWS"],
    "video_frames": 103742,
}


def sort_by_avg_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the leaderboard by average rank across all metrics.

    Each metric is ranked descending (1 = best); missing values sort to the
    bottom of that metric. A model's score is the mean rank across metrics —
    lower is better. Ties in a given metric share a rank, and the next
    distinct score receives the next consecutive rank (dense ranking). This
    avoids large, artificial rank gaps when many rounded scores are tied.

    Deterministic tiebreakers (applied in order when avg-rank is equal):
      1. Number of metrics where the model is ranked #1 (dominance) —
         a model that wins 6 metrics outright beats one that wins 3,
         even at the same mean rank.
      2. Sum of per-metric ranks (catches sub-ties from the mean).
      3. Sum of normalized raw metric values (slight-edge winner).
      4. Model name alphabetical — guarantees full determinism so the
         same model gets the same rank in any table that contains the
         same competitors, regardless of input row order.
    """
    if df.empty:
        return df.reset_index(drop=True)

    metric_keys = [k for k in METRICS.keys() if k in df.columns]
    if not metric_keys:
        return df.reset_index(drop=True)

    ranks = pd.DataFrame(index=df.index)
    for m in metric_keys:
        col = pd.to_numeric(df[m], errors="coerce")
        # Rank descending using consecutive ranks for distinct scores. Scores
        # are stored to three decimals, so competition ranking (method="min")
        # can otherwise create large gaps after a common tied value.
        ranks[m] = col.rank(ascending=False, method="dense", na_option="bottom")

    df = df.copy()
    df["_avg_rank"] = ranks.mean(axis=1)
    # Tiebreaker 1: count of metrics where this model is rank 1 (dominance).
    # A model that's first on 6 metrics beats one that's first on 3, even if
    # their mean ranks are equal — matches the intuition that "dominant on
    # many metrics" should win over "narrowly better on a few".
    df["_n_wins"] = -(ranks == 1).sum(axis=1)  # negate so ascending sort = more wins first
    # Tiebreaker 2: sum of per-metric ranks (catches sub-ties).
    df["_sum_rank"] = ranks.sum(axis=1)
    # Tiebreaker 3: sum of normalized raw scores (slight-edge winner).
    normed = pd.DataFrame(index=df.index)
    for m in metric_keys:
        col = pd.to_numeric(df[m], errors="coerce")
        col_max = col.max()
        normed[m] = (col / col_max) if (col_max and col_max > 0) else 0.0
    df["_sum_norm"] = -normed.sum(axis=1)  # negate so ascending sort = higher score first
    # Tiebreaker 4: model name alphabetical (final fallback for full determinism).
    df["_name_key"] = df["model_name"].astype(str) if "model_name" in df.columns else ""

    df = df.sort_values(
        ["_avg_rank", "_n_wins", "_sum_rank", "_sum_norm", "_name_key"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    df = df.drop(columns=["_avg_rank", "_n_wins", "_sum_rank", "_sum_norm", "_name_key"])
    # Reassign the displayed rank column to match the new ordering — otherwise
    # stale values from the stored JSON (or an older sort) drive the medal badges.
    df["rank"] = range(1, len(df) + 1)
    return df


def load_leaderboard() -> pd.DataFrame:
    """
    Load leaderboard from private HuggingFace repo.
    Falls back to local file for development.
    """
    try:
        # Try loading from private repo first
        token = os.environ.get('HF_TOKEN')
        if token:
            print("⏳ Downloading leaderboard from private repository...")
            try:
                leaderboard_file = hf_hub_download(
                    repo_id="UII-AI/MedVidBench-GroundTruth",
                    filename="leaderboard.json",
                    repo_type="dataset",
                    token=token,
                    cache_dir="./cache"
                )

                with open(leaderboard_file, 'r') as f:
                    data = json.load(f)

                if data:
                    df = pd.DataFrame(data)

                    # Remove any 'average' column from old leaderboard format
                    if 'average' in df.columns:
                        df = df.drop('average', axis=1)

                    # Sort by average rank within this table only — each leaderboard
                    # is ranked independently against its own rows so the displayed
                    # rank column is sequential (1, 2, 3, ... with no gaps).
                    if 'cvs_acc' in df.columns:
                        df = sort_by_avg_rank(df)

                    print(f"✓ Loaded leaderboard from private repo: {len(df)} entries")
                    return df
            except Exception as e:
                print(f"⚠️  Could not load leaderboard from private repo: {e}")
                print("   Using local fallback...")
    except Exception:
        pass

    # Fallback to local file
    if LEADERBOARD_FILE.exists():
        with open(LEADERBOARD_FILE, 'r') as f:
            data = json.load(f)
        if data:
            df = pd.DataFrame(data)

            # Remove any 'average' column from old leaderboard format
            if 'average' in df.columns:
                df = df.drop('average', axis=1)

            # Sort by avg-rank within this table only (see note in repo loader).
            if 'cvs_acc' in df.columns:
                df = sort_by_avg_rank(df)

            print(f"✓ Loaded leaderboard from local file: {len(df)} entries")
            return df

    # Return empty dataframe with correct structure (no average column)
    print("📋 No existing leaderboard found, starting fresh")
    columns = (
        ["rank", "model_name", "organization", "team_name", "authors"]
        + list(METRICS.keys())
        + ["date", "contact", "model_url", "is_eccv"]
    )
    return pd.DataFrame(columns=columns)


def save_leaderboard(df: pd.DataFrame):
    """
    Save leaderboard to both local file and private HuggingFace repo.
    This ensures persistence across app updates on HuggingFace Spaces.
    """
    # Add rank column
    df['rank'] = range(1, len(df) + 1)

    # Save to local JSON first
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(df.to_dict('records'), f, indent=2)

    print(f"✓ Saved leaderboard locally: {len(df)} entries")

    # Upload to private HuggingFace repo
    try:
        token = os.environ.get('HF_TOKEN')
        if not token:
            print("⚠️  HF_TOKEN not found in environment")
            print("   Set HF_TOKEN secret in Space settings to enable repo sync")
            print("   Leaderboard saved locally only (will not persist across restarts)")
            return

        print("⏳ Uploading leaderboard to private repository...")
        print(f"   Target: UII-AI/MedVidBench-GroundTruth/leaderboard.json")
        print(f"   Entries: {len(df)}")

        api = HfApi()

        # Upload with detailed error handling
        result = api.upload_file(
            path_or_fileobj=str(LEADERBOARD_FILE),
            path_in_repo="leaderboard.json",
            repo_id="UII-AI/MedVidBench-GroundTruth",
            repo_type="dataset",
            token=token,
            commit_message=f"Update leaderboard: {len(df)} entries ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
        )

        print(f"✓ Successfully uploaded leaderboard to private repo")
        print(f"   Commit URL: {result}")

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed to upload leaderboard to private repo")
        print(f"   Error: {error_msg}")

        # Provide specific guidance based on error type
        if "401" in error_msg or "Unauthorized" in error_msg:
            print("   → Issue: Invalid or expired token")
            print("   → Fix: Regenerate HF_TOKEN with write permission")
        elif "404" in error_msg or "Not Found" in error_msg:
            print("   → Issue: Repository not found")
            print("   → Fix: Create UII-AI/MedVidBench-GroundTruth repo")
        elif "403" in error_msg or "Forbidden" in error_msg:
            print("   → Issue: Token lacks write permission")
            print("   → Fix: Use token with write access to dataset")
        else:
            print(f"   → Check HuggingFace status and repo permissions")

        print("   ⚠️  Leaderboard saved locally only (will not persist)")


def load_official_leaderboard() -> pd.DataFrame:
    """
    Load official (verified) leaderboard from private HuggingFace repo.
    Falls back to local file for development.
    """
    try:
        token = os.environ.get('HF_TOKEN')
        if token:
            try:
                official_file = hf_hub_download(
                    repo_id="UII-AI/MedVidBench-GroundTruth",
                    filename="official_leaderboard.json",
                    repo_type="dataset",
                    token=token,
                    cache_dir="./cache"
                )
                with open(official_file, 'r') as f:
                    data = json.load(f)
                if data:
                    df = pd.DataFrame(data)
                    if 'cvs_acc' in df.columns:
                        df = sort_by_avg_rank(df)
                    print(f"✓ Loaded official leaderboard from private repo: {len(df)} entries")
                    return df
            except Exception as e:
                print(f"⚠️  Could not load official leaderboard from private repo: {e}")
    except Exception:
        pass

    # Fallback to local file
    if OFFICIAL_LEADERBOARD_FILE.exists():
        with open(OFFICIAL_LEADERBOARD_FILE, 'r') as f:
            data = json.load(f)
        if data:
            df = pd.DataFrame(data)
            if 'cvs_acc' in df.columns:
                df = sort_by_avg_rank(df)
            print(f"✓ Loaded official leaderboard from local file: {len(df)} entries")
            return df

    print("📋 No official leaderboard found, starting fresh")
    columns = ["rank", "model_name", "organization"] + list(METRICS.keys()) + ["date", "verified_date", "contact"]
    return pd.DataFrame(columns=columns)


def save_official_leaderboard(df: pd.DataFrame):
    """
    Save official leaderboard to both local file and private HuggingFace repo.
    """
    df['rank'] = range(1, len(df) + 1)

    with open(OFFICIAL_LEADERBOARD_FILE, 'w') as f:
        json.dump(df.to_dict('records'), f, indent=2)

    print(f"✓ Saved official leaderboard locally: {len(df)} entries")

    try:
        token = os.environ.get('HF_TOKEN')
        if not token:
            print("⚠️  HF_TOKEN not found, official leaderboard saved locally only")
            return

        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(OFFICIAL_LEADERBOARD_FILE),
            path_in_repo="official_leaderboard.json",
            repo_id="UII-AI/MedVidBench-GroundTruth",
            repo_type="dataset",
            token=token,
            commit_message=f"Update official leaderboard: {len(df)} entries ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
        )
        print(f"✓ Uploaded official leaderboard to private repo")

    except Exception as e:
        print(f"❌ Failed to upload official leaderboard: {e}")
        print("   Saved locally only")


def add_to_official_leaderboard(model_name: str, organization: str, metrics: Dict[str, float],
                                 contact: str = "") -> Tuple[bool, str]:
    """
    Add a verified model to the official leaderboard.
    Called by admin after verifying model via API access.
    """
    df = load_official_leaderboard()

    if model_name in df['model_name'].values:
        return False, f"Model '{model_name}' already exists in official leaderboard"

    new_entry = {
        "model_name": model_name,
        "organization": organization,
        **{metric: round(metrics.get(metric, 0.0), 3) for metric in METRICS.keys()},
        "date": datetime.now().strftime("%Y-%m-%d"),
        "verified_date": datetime.now().strftime("%Y-%m-%d"),
        "contact": contact,
    }

    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df = sort_by_avg_rank(df)
    save_official_leaderboard(df)

    return True, f"✓ Added '{model_name}' to official leaderboard (rank #{df[df['model_name'] == model_name].index[0] + 1})"


def remove_from_official_leaderboard(model_name: str) -> Tuple[bool, str]:
    """Remove a model from the official leaderboard."""
    df = load_official_leaderboard()

    if df.empty or model_name not in df['model_name'].values:
        return False, f"Model '{model_name}' not found in official leaderboard"

    df = df[df['model_name'] != model_name].reset_index(drop=True)
    save_official_leaderboard(df)
    return True, f"✓ Removed '{model_name}' from official leaderboard"


def promote_to_official(model_name: str) -> Tuple[bool, str]:
    """
    Promote a model from community submissions to the official leaderboard.
    Copies all metrics from the community leaderboard entry.
    """
    community_df = load_leaderboard()

    if community_df.empty or model_name not in community_df['model_name'].values:
        return False, f"Model '{model_name}' not found in community leaderboard"

    official_df = load_official_leaderboard()
    if model_name in official_df['model_name'].values:
        return False, f"Model '{model_name}' already exists in official leaderboard"

    row = community_df[community_df['model_name'] == model_name].iloc[0]
    metrics = {k: row.get(k, 0.0) for k in METRICS.keys()}

    return add_to_official_leaderboard(
        model_name=model_name,
        organization=row.get('organization', ''),
        metrics=metrics,
        contact=row.get('contact', ''),
    )


def format_official_leaderboard_html(df: pd.DataFrame) -> str:
    """Render official leaderboard as a styled HTML table with verified badge."""
    if df.empty:
        return "<p style='text-align:center; color:#6b7280; padding:2rem;'>No verified models yet. Top community submissions will be verified and added here.</p>"

    metric_keys = [k for k in METRICS.keys() if k in df.columns]
    display_cols = ["rank", "model_name", "organization"] + metric_keys + ["verified_date"]

    header_map = {
        "rank": "Rank", "model_name": "Model", "organization": "Team",
        "verified_date": "Verified",
    }
    for k in metric_keys:
        header_map[k] = METRICS[k]["name"]

    rank_styles = {
        1: "background:#fef2f2; font-weight:600;",
        2: "background:#fffbeb; font-weight:600;",
        3: "background:#f0fdf4; font-weight:600;",
    }
    rank_badges = {1: "🥇", 2: "🥈", 3: "🥉"}

    metric_best = {}
    metric_second = {}
    for k in metric_keys:
        vals = df[k].dropna().astype(float)
        if len(vals) >= 1:
            sorted_vals = vals.sort_values(ascending=False).unique()
            metric_best[k] = sorted_vals[0] if len(sorted_vals) >= 1 else None
            metric_second[k] = sorted_vals[1] if len(sorted_vals) >= 2 else None

    html = """<style>
.off-table-wrap { background:#ffffff; color:#111827; border-radius:6px; }
.off-table { width:100%; border-collapse:collapse; font-size:0.85rem; font-family:system-ui,-apple-system,sans-serif; background:#ffffff; color:#111827; }
.off-table th { background:#1e3a5f !important; color:#ffffff !important; padding:8px 6px; text-align:center; border-bottom:2px solid #1e3a5f; font-size:0.78rem; position:sticky; top:0; }
.off-table td { padding:6px; text-align:center; border-bottom:1px solid #e5e7eb; background:#ffffff; color:#111827; }
.off-table td * { color:inherit !important; }
.off-table td .eccv-badge { color:#ffd86b !important; }
.off-table tr:hover td { background:#f0f7ff; }
.off-table tr[data-rank="1"] td { background:#fef2f2; font-weight:600; }
.off-table tr[data-rank="2"] td { background:#fffbeb; font-weight:600; }
.off-table tr[data-rank="3"] td { background:#f0fdf4; font-weight:600; }
.off-table .model-col { text-align:left; font-weight:500; min-width:230px; white-space:nowrap; }
.off-table .model-col a { color:#1e3a5f !important; }
.off-table .org-col { text-align:left; color:#6b7280; font-size:0.8rem; white-space:nowrap; }
.off-table .org-col * { color:#6b7280 !important; }
.off-table .best-cell, .off-table td.best-cell, .off-table td.best-cell * { color:#b91c1c !important; font-weight:700; }
.off-table .second-cell, .off-table td.second-cell, .off-table td.second-cell * { color:#b45309 !important; font-weight:600; }
@media (prefers-color-scheme: dark) {
  .off-table-wrap { background:#111827; color:#e5e7eb; }
  .off-table { background:#111827; color:#e5e7eb; }
  .off-table th { background:#0f1e33 !important; color:#e5e7eb !important; border-bottom-color:#0f1e33; }
  .off-table td { background:#111827; color:#e5e7eb; border-bottom-color:#1f2937; }
  .off-table tr:hover td { background:#1e293b; }
  .off-table tr[data-rank="1"] td { background:#3b1f23; }
  .off-table tr[data-rank="2"] td { background:#3b2e14; }
  .off-table tr[data-rank="3"] td { background:#132b1d; }
  .off-table .model-col a { color:#93c5fd !important; border-bottom-color:#93c5fd !important; }
  .off-table .org-col, .off-table .org-col * { color:#9ca3af !important; }
  .off-table .best-cell, .off-table td.best-cell, .off-table td.best-cell * { color:#fca5a5 !important; }
  .off-table .second-cell, .off-table td.second-cell, .off-table td.second-cell * { color:#fcd34d !important; }
}
</style>
<div class="off-table-wrap" style="overflow-x:auto; max-height:600px; overflow-y:auto;">
<table class="off-table">
<thead><tr>"""

    for col in display_cols:
        html += f"<th>{header_map.get(col, col)}</th>"
    html += "</tr></thead>\n<tbody>"

    for _, row in df.iterrows():
        rank = int(row.get('rank', 0))
        html += f'<tr data-rank="{rank}">'

        for col in display_cols:
            val = row.get(col, "")

            if col == "rank":
                badge = rank_badges.get(rank, "")
                html += f'<td style="white-space:nowrap;">{badge}{rank}</td>'
            elif col == "model_name":
                url = row.get("model_url", "")
                url = "" if (url is None or (isinstance(url, float) and pd.isna(url))) else str(url).strip()
                name_html = f'<a href="{url}" target="_blank" rel="noopener" style="color:#1e3a5f; text-decoration:none; border-bottom:1px dotted #1e3a5f;">{val}</a>' if url else f'{val}'
                is_eccv = row.get("is_eccv", False)
                if isinstance(is_eccv, float) and pd.isna(is_eccv):
                    is_eccv = False
                eccv_badge = (
                    ' <span class="eccv-badge" title="MedVidU @ ECCV 2026 Challenge submission" '
                    'style="display:inline-block; margin-left:6px; padding:1px 6px; '
                    'background:#1e3a5f; color:#ffd86b; font-size:0.68rem; font-weight:700; '
                    'border-radius:4px; vertical-align:middle;">ECCV&#39;26</span>'
                    if is_eccv else ""
                )
                html += f'<td class="model-col">{name_html}{eccv_badge}</td>'
            elif col == "organization":
                status = row.get("status", "")
                status_badge = {"api_verified": "&#9989; "}.get(status, "")
                html += f'<td class="org-col">{status_badge}{val}</td>'
            elif col == "verified_date":
                html += f'<td style="color:#059669; font-size:0.78rem;">{val}</td>'
            elif col in metric_keys:
                fval = round(float(val), 3) if pd.notna(val) else 0.0
                cell_class = ""
                if metric_best.get(col) is not None and fval == metric_best[col]:
                    cell_class = "best-cell"
                elif metric_second.get(col) is not None and fval == metric_second[col]:
                    cell_class = "second-cell"
                html += f'<td class="{cell_class}">{fval:.3f}</td>'
            else:
                html += f'<td>{val}</td>'

        html += "</tr>\n"

    html += "</tbody></table></div>"

    html += """<p style="text-align:center; margin-top:0.5rem; font-size:0.78rem; color:#6b7280;">
    <span style="color:#b91c1c; font-weight:700;">&#9632; Best</span> &nbsp;
    <span style="color:#b45309; font-weight:600;">&#9632; 2nd Best</span> &nbsp;&nbsp;
    &#9989; = User submission verified by maintainers via model API &nbsp;&nbsp;
    <span style="display:inline-block; padding:1px 6px; background:#1e3a5f; color:#ffd86b; font-size:0.68rem; font-weight:700; border-radius:4px;">ECCV&#39;26</span>
    = <a href="https://medvidu.github.io/" target="_blank" rel="noopener" style="color:#1e3a5f;">MedVidU @ ECCV 2026 Challenge</a> entry &nbsp;&nbsp;
    * = off-the-shelf models
    </p>"""

    return html


def backup_results_to_repo(model_name: str, results_dir: Path):
    """
    Backup full evaluation results to private HuggingFace repo.
    This stores detailed evaluation logs for each submission.
    """
    try:
        token = os.environ.get('HF_TOKEN')
        if not token:
            return

        # Check if results directory exists
        if not results_dir.exists():
            return

        print(f"⏳ Backing up results for {model_name} to private repository...")

        api = HfApi()

        # Upload eval_output.txt if exists
        eval_output = results_dir / "eval_output.txt"
        if eval_output.exists():
            api.upload_file(
                path_or_fileobj=str(eval_output),
                path_in_repo=f"results/{model_name}/eval_output.txt",
                repo_id="UII-AI/MedVidBench-GroundTruth",
                repo_type="dataset",
                token=token,
                commit_message=f"Backup results for {model_name}"
            )

        # Upload input.json if exists (the submitted predictions)
        input_file = results_dir / "input.json"
        if input_file.exists():
            api.upload_file(
                path_or_fileobj=str(input_file),
                path_in_repo=f"results/{model_name}/input.json",
                repo_id="UII-AI/MedVidBench-GroundTruth",
                repo_type="dataset",
                token=token,
                commit_message=f"Backup predictions for {model_name}"
            )

        print(f"✓ Backed up results for {model_name}")

    except Exception as e:
        print(f"⚠️  Failed to backup results: {e}")


# ============================================================================
# Admin Functions
# ============================================================================

def check_admin_password(password: str) -> bool:
    """
    Check if provided password matches admin password.
    Admin password is set via ADMIN_PASSWORD environment variable.
    """
    admin_password = os.environ.get('ADMIN_PASSWORD', '')

    if not admin_password:
        # If no admin password set, use a default (should be changed in production)
        admin_password = 'admin-2025'

    return password == admin_password


def delete_model_submission(model_name: str) -> Tuple[bool, str]:
    """
    Delete a model submission from leaderboard and cleanup associated files.

    Args:
        model_name: Name of the model to delete

    Returns:
        (success, message)
    """
    try:
        # Load current leaderboard
        df = load_leaderboard()

        if df.empty:
            return False, "Leaderboard is empty"

        # Check if model exists
        if model_name not in df['model_name'].values:
            return False, f"Model '{model_name}' not found in leaderboard"

        # Get model info before deletion
        model_row = df[df['model_name'] == model_name].iloc[0]
        organization = model_row.get('organization', 'Unknown')
        date = model_row.get('date', 'Unknown')

        # Remove from leaderboard
        df = df[df['model_name'] != model_name].reset_index(drop=True)
        save_leaderboard(df)

        # Cleanup associated files
        model_dir_name = model_name.replace(" ", "_")
        results_dir = RESULTS_DIR / model_dir_name
        submissions_dir = SUBMISSIONS_DIR / model_dir_name

        cleanup_info = []

        # Remove results directory
        if results_dir.exists():
            shutil.rmtree(results_dir)
            cleanup_info.append(f"Removed results: {results_dir}")

        # Remove submissions directory
        if submissions_dir.exists():
            shutil.rmtree(submissions_dir)
            cleanup_info.append(f"Removed submissions: {submissions_dir}")

        message = f"✓ Successfully deleted model '{model_name}'\n"
        message += f"  Organization: {organization}\n"
        message += f"  Date: {date}\n\n"
        if cleanup_info:
            message += "Cleaned up:\n" + "\n".join(f"  - {info}" for info in cleanup_info)

        return True, message

    except Exception as e:
        return False, f"Error deleting model: {str(e)}"


def get_leaderboard_for_admin() -> pd.DataFrame:
    """Get leaderboard data formatted for admin view."""
    df = load_leaderboard()

    if df.empty:
        return pd.DataFrame(columns=["rank", "model_name", "organization", "date", "contact"])

    # Select key columns for admin view
    admin_cols = ["rank", "model_name", "organization", "date", "contact"]
    available_cols = [col for col in admin_cols if col in df.columns]

    return df[available_cols]


def detect_evaluation_output_format(file_path: str) -> Tuple[bool, str]:
    """
    Detect if uploaded file is pre-processed evaluation output with LLM judge scores.

    Expected evaluation output format:
    {
        "model_name": "...",
        "evaluated_samples": [
            {
                "sample_id": "...",
                "dataset": "...",
                "evaluations": {
                    "0.3": [{"prediction": "...", "ground_truth": "...",
                             "llm_judge": {"R2": 4, "R3": 2, ...}}],
                    "0.5": [...],
                    "0.7": [...]
                }
            }
        ],
        "aggregated_results": {...}
    }

    Returns:
        (is_evaluation_output, message)

    Note: Pre-processed files typically only have LLM judge scores for captioning tasks (DVC, VS, RC).
          Other metrics (TAL, STG, NAP, SA, CVS) still need to be calculated from raw inference results.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Check for evaluation output structure
        if isinstance(data, dict):
            has_model_name = "model_name" in data
            has_evaluated_samples = "evaluated_samples" in data
            has_aggregated_results = "aggregated_results" in data

            if has_model_name and has_evaluated_samples and has_aggregated_results:
                # Verify structure by checking first evaluated sample
                if len(data["evaluated_samples"]) > 0:
                    sample = data["evaluated_samples"][0]
                    if "evaluations" in sample and isinstance(sample["evaluations"], dict):
                        return True, "✓ Detected pre-processed evaluation output with LLM judge scores (captioning tasks only)"

        return False, "Not an evaluation output file"

    except Exception as e:
        return False, f"Error detecting format: {str(e)}"


def check_for_precomputed_llm_scores(file_path: str) -> Tuple[bool, Optional[Dict]]:
    """
    Check if the results file has pre-computed LLM judge scores in struc_info.

    Returns:
        (has_precomputed_scores, llm_score_dict or None)
        llm_score_dict format: {'dvc': score, 'vs': score, 'rc': score}
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, dict):
            records = list(data.values())
        elif isinstance(data, list):
            records = data
        else:
            return False, None

        if len(records) == 0:
            return False, None

        # Check if records have struc_info with llm_judge scores
        has_llm_scores = False
        for record in records[:10]:  # Check first 10 samples
            if "struc_info" in record:
                struc_info = record.get("struc_info", [])
                if isinstance(struc_info, list) and len(struc_info) > 0:
                    for item in struc_info:
                        if isinstance(item, dict) and "llm_judge" in item:
                            has_llm_scores = True
                            break
            if has_llm_scores:
                break

        return has_llm_scores, None  # Return None for score dict as we'll get it from evaluation

    except Exception as e:
        return False, None


def validate_results_file(file_path: str) -> Tuple[bool, str, bool]:
    """
    Validate uploaded file - accepts both prediction-only and merged formats.

    Expected format for predictions (preferred):
    [
        {
            "id": "video_id&&start&&end&&fps",
            "qa_type": "tal/stg/next_action/dvc/vs/rc/skill_assessment/cvs_assessment",
            "prediction": "Model's answer..."
        },
        ...
    ]

    Also accepts merged format (for testing):
    {
        "0": {
            "metadata": {...},
            "qa_type": "tal",
            "question": "...",
            "answer": "...",
            "gnd": "...",
            "struc_info": [...]
        },
        ...
    }

    Returns:
        (valid, message, has_precomputed_llm_scores)
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Handle both list and dict formats
        if isinstance(data, dict):
            records = list(data.values())
        elif isinstance(data, list):
            records = data
        else:
            return False, f"Invalid format: expected list or dict, got {type(data)}", False

        if len(records) == 0:
            return False, "Empty predictions file", False

        # Check first record to detect format
        sample = records[0]

        # Format 1: Prediction-only (id, qa_type, prediction)
        is_prediction_only = "id" in sample and "prediction" in sample

        # Format 2: Merged (metadata, question, answer, gnd, struc_info)
        is_merged = "metadata" in sample and "question" in sample and "answer" in sample

        if is_prediction_only:
            # Validate prediction-only format
            if "qa_type" not in sample:
                return False, "Missing required field: 'qa_type'", False

            # Check qa_type is valid
            valid_qa_types = ["tal", "stg", "next_action", "dense_captioning", "video_summary", "region_caption",
                              "skill_assessment", "cvs_assessment"]
            qa_type = sample.get("qa_type", "")
            if not any(valid in qa_type for valid in valid_qa_types):
                return False, f"Invalid qa_type: {qa_type}", False

            # Check if file has reasonable number of samples (relaxed requirement)
            if len(records) < 100:
                return False, f"Too few samples ({len(records)}). Need at least 100 samples.", False

            return True, f"✓ Valid predictions file (prediction-only format) with {len(records)} samples", False

        elif is_merged:
            # Validate merged format
            if "qa_type" not in sample:
                return False, "Missing required field: 'qa_type'", False

            # Check if file has reasonable number of samples (relaxed requirement)
            if len(records) < 100:
                return False, f"Too few samples ({len(records)}). Need at least 100 samples.", False

            return True, f"✓ Valid predictions file (merged format) with {len(records)} samples", False

        else:
            return False, "Invalid format: Must be either prediction-only (id, qa_type, prediction) or merged format (metadata, question, answer)", False

    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}", False
    except Exception as e:
        return False, f"Error validating file: {str(e)}", False


def extract_metrics_from_evaluation_output(file_path: str) -> Tuple[bool, Dict, str]:
    """
    Extract metrics directly from pre-processed evaluation output.

    Returns:
        (success, metrics_dict, message)
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict) or "aggregated_results" not in data:
            return False, {}, "Invalid evaluation output structure"

        aggregated = data["aggregated_results"]
        metrics = {}

        # Extract metrics from aggregated results
        # The structure varies by task, so we need to handle each carefully

        # TAL: Extract mIoU@0.3 and mIoU@0.5
        if "tal" in aggregated or "TAL" in aggregated:
            tal_results = aggregated.get("tal", aggregated.get("TAL", {}))
            if "overall" in tal_results:
                overall = tal_results["overall"]
                metrics["tag_miou_03"] = overall.get("meanIoU@0.3", 0.0)
                metrics["tag_miou_05"] = overall.get("meanIoU@0.5", 0.0)

        # STG: Extract mIoU
        if "stg" in aggregated or "STG" in aggregated:
            stg_results = aggregated.get("stg", aggregated.get("STG", {}))
            if "overall" in stg_results:
                overall = stg_results["overall"]
                metrics["stg_miou"] = overall.get("mean_iou", 0.0)

        # Next Action: Extract accuracy
        if "next_action" in aggregated or "NEXT_ACTION" in aggregated:
            nap_results = aggregated.get("next_action", aggregated.get("NEXT_ACTION", {}))
            if "overall" in nap_results:
                overall = nap_results["overall"]
                metrics["nap_acc"] = overall.get("accuracy", 0.0)

        # DVC: Extract caption_score and temporal_f1
        if "dvc" in aggregated or "DVC" in aggregated or "dense_captioning" in aggregated:
            dvc_results = aggregated.get("dvc", aggregated.get("DVC", aggregated.get("dense_captioning", {})))
            if "overall" in dvc_results:
                overall = dvc_results["overall"]
                metrics["dvc_llm"] = overall.get("caption_score", 0.0)
                metrics["dvc_f1"] = overall.get("temporal_f1", 0.0)

        # VS: Extract LLM score
        if "vs" in aggregated or "VS" in aggregated or "video_summary" in aggregated:
            vs_results = aggregated.get("vs", aggregated.get("VS", aggregated.get("video_summary", {})))
            if "overall" in vs_results:
                overall = vs_results["overall"]
                # Try multiple possible field names
                metrics["vs_llm"] = overall.get("score", overall.get("average_score", overall.get("caption_score", 0.0)))

        # RC: Extract LLM score
        if "rc" in aggregated or "RC" in aggregated or "region_caption" in aggregated:
            rc_results = aggregated.get("rc", aggregated.get("RC", aggregated.get("region_caption", {})))
            if "overall" in rc_results:
                overall = rc_results["overall"]
                # Try multiple possible field names
                metrics["rc_llm"] = overall.get("score", overall.get("average_score", overall.get("caption_score", 0.0)))

        # Skill Assessment: Extract aspect_balanced_accuracy (matches run_eval.sh)
        if "skill_assessment" in aggregated or "SKILL" in aggregated:
            sa_results = aggregated.get("skill_assessment", aggregated.get("SKILL", {}))
            if "overall" in sa_results:
                overall = sa_results["overall"]
                metrics["sa_acc"] = overall.get("aspect_balanced_accuracy", overall.get("accuracy", 0.0))

        # CVS Assessment: Extract component_balanced_accuracy (matches run_eval.sh)
        if "cvs_assessment" in aggregated or "CVS" in aggregated:
            cvs_results = aggregated.get("cvs_assessment", aggregated.get("CVS", {}))
            if "overall" in cvs_results:
                overall = cvs_results["overall"]
                metrics["cvs_acc"] = overall.get("component_balanced_accuracy", overall.get("accuracy", 0.0))

        # Check if we got all 10 metrics
        if len(metrics) < 10:
            missing = [m for m in METRICS.keys() if m not in metrics]
            return False, metrics, f"Incomplete metrics extracted. Missing: {missing}"

        return True, metrics, "✓ Metrics extracted from pre-processed evaluation output"

    except Exception as e:
        return False, {}, f"Error extracting metrics: {str(e)}"


def run_evaluation(results_file: str, model_name: str, has_precomputed_llm: bool = False,
                   log_callback=None) -> Tuple[bool, Dict, str]:
    """
    Run evaluation using evaluate_predictions.py wrapper.

    Handles both prediction-only and merged formats automatically.

    Args:
        results_file: Path to predictions JSON (either format)
        model_name: Name of the model
        has_precomputed_llm: Not used (kept for compatibility)
        log_callback: Optional callback function(line) to stream logs

    Returns:
        (success, metrics_dict, message)
    """
    try:
        # Create output directory for this submission
        output_dir = RESULTS_DIR / model_name.replace(" ", "_")
        output_dir.mkdir(exist_ok=True)

        # Save user file
        input_file = output_dir / "input.json"
        shutil.copy(results_file, input_file)

        # Use evaluate_predictions.py wrapper which handles both formats
        eval_wrapper = Path("evaluation/evaluate_predictions.py")

        cmd = [
            sys.executable,
            str(eval_wrapper),
            str(input_file),
            "--grouping", "overall",
            "--ground-truth", "data/ground_truth.json"
        ]

        print("=" * 60)
        print("RUNNING EVALUATION")
        print("=" * 60)
        print(f"Command: {' '.join(cmd)}")

        # Run with real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Capture output while streaming to callback
        output_lines = []
        for line in process.stdout:
            line = line.rstrip()
            output_lines.append(line)
            print(line)  # Print to server logs

            # Stream to callback if provided
            if log_callback and line.strip():
                log_callback(line)

        # Wait for completion
        process.wait(timeout=600)

        if process.returncode != 0:
            full_output = '\n'.join(output_lines)
            return False, {}, f"Evaluation failed (exit code {process.returncode})"

        # Parse output to extract metrics
        full_output = '\n'.join(output_lines)
        metrics = parse_evaluation_output(full_output)

        # Save evaluation output
        with open(output_dir / "eval_output.txt", 'w') as f:
            f.write(full_output)

        print(f"✓ Evaluation completed")
        print("=" * 60)

        return True, metrics, "✓ Evaluation completed successfully"

    except subprocess.TimeoutExpired:
        if process:
            process.kill()
        return False, {}, "Evaluation timed out (>10 minutes)"
    except Exception as e:
        return False, {}, f"Error running evaluation: {str(e)}"


def parse_evaluation_output(output: str) -> Dict[str, float]:
    """
    Parse evaluation output to extract 10 metrics.

    Returns dict with keys:
        cvs_acc, nap_acc, sa_acc, stg_miou,
        tag_miou_03, tag_miou_05, dvc_llm, dvc_f1, vs_llm, rc_llm
    """
    metrics = {}

    # Prefer the clean LEADERBOARD METRICS SUMMARY block if present — it's the
    # canonical, deduplicated output from evaluate_all.py. Falling through to
    # the full log causes per-dataset lines to overwrite overall metrics.
    if "LEADERBOARD METRICS SUMMARY" in output and "END LEADERBOARD METRICS SUMMARY" in output:
        start = output.index("LEADERBOARD METRICS SUMMARY")
        end = output.index("END LEADERBOARD METRICS SUMMARY")
        output = output[start:end]

    lines = output.split('\n')
    current_task = None
    current_iou_section = None  # Track IoU_0.3 or IoU_0.5 sections for TAL

    for i, line in enumerate(lines):
        line = line.strip()

        # Detect task headers
        # NOTE: Order matters — check CVS before VS (since "CVS" contains "VS")
        if ("CVS" in line and "Overall" in line) or "CVS Assessment" in line:
            current_task = "cvs_assessment"
        elif ("SKILL" in line and "Overall" in line) or "Skill Assessment" in line:
            current_task = "skill_assessment"
        elif "TAL" in line and "Overall" in line:
            current_task = "tal"
        elif "STG" in line and "Overall" in line:
            current_task = "stg"
        elif ("NEXT_ACTION" in line and "Overall" in line) or "Next Action" in line:
            current_task = "next_action"
        elif ("DVC" in line and "Overall" in line) or "Dense Video Captioning" in line:
            current_task = "dvc"
        elif ("RC" in line and "Overall" in line) or "Region Caption" in line:
            current_task = "rc"
        elif ("VS" in line and "Overall" in line) or "Video Summary" in line:
            current_task = "vs"

        # Detect IoU sections for TAL (new format)
        if current_task == "tal":
            if "IoU_0.3:" in line:
                current_iou_section = "0.3"
            elif "IoU_0.5:" in line:
                current_iou_section = "0.5"

        # Extract metrics based on task
        if current_task:
            # TAL: Extract both mIoU@0.3 and mIoU@0.5
            if current_task == "tal":
                # Old format: meanIoU@0.3 or mIoU@0.3
                if "meanIoU@0.3" in line or "mIoU@0.3" in line:
                    try:
                        value = float(line.split(":")[-1].strip())
                        metrics["tag_miou_03"] = value
                    except:
                        pass
                if "meanIoU@0.5" in line or "mIoU@0.5" in line:
                    try:
                        value = float(line.split(":")[-1].strip())
                        metrics["tag_miou_05"] = value
                    except:
                        pass
                # New format: IoU_0.3: section with meanIoU:
                if current_iou_section and "meanIoU:" in line and "meanIoU@" not in line:
                    try:
                        value = float(line.split(":")[-1].strip())
                        if current_iou_section == "0.3":
                            metrics["tag_miou_03"] = value
                        elif current_iou_section == "0.5":
                            metrics["tag_miou_05"] = value
                    except:
                        pass

            # STG: Extract mIoU
            elif current_task == "stg" and ("mean_iou" in line.lower() or "miou" in line.lower()):
                try:
                    value = float(line.split(":")[-1].strip())
                    metrics["stg_miou"] = value
                except:
                    pass

            # Next Action: Extract accuracy (strip trailing "(count/total)" if present)
            elif current_task == "next_action" and "accuracy" in line.lower():
                try:
                    value = float(line.split(":")[-1].split("(")[0].strip())
                    metrics["nap_acc"] = value
                except:
                    pass

            # DVC: Extract both caption_score and temporal_f1
            elif current_task == "dvc":
                if "caption_score" in line.lower() or "caption score" in line.lower():
                    try:
                        value = float(line.split(":")[-1].strip())
                        metrics["dvc_llm"] = value
                    except:
                        pass
                if "temporal_f1" in line.lower() or "temporal f1" in line.lower():
                    try:
                        value = float(line.split(":")[-1].strip())
                        metrics["dvc_f1"] = value
                    except:
                        pass

            # VS: Extract LLM score
            elif current_task == "vs" and ("score" in line.lower() or "average" in line.lower()):
                try:
                    val_str = line.split(":")[-1].strip().split("(")[0].strip()
                    metrics["vs_llm"] = float(val_str)
                except:
                    pass

            # RC: Extract LLM score
            elif current_task == "rc" and ("score" in line.lower() or "average" in line.lower()):
                try:
                    val_str = line.split(":")[-1].strip().split("(")[0].strip()
                    metrics["rc_llm"] = float(val_str)
                except:
                    pass

            # Skill Assessment: Extract aspect_balanced_accuracy (matches run_eval.sh)
            elif current_task == "skill_assessment" and "aspect_balanced_accuracy" in line.lower():
                try:
                    value = float(line.split(":")[-1].split("(")[0].strip())
                    metrics["sa_acc"] = value
                except:
                    pass

            # CVS Assessment: Extract component_balanced_accuracy (matches run_eval.sh)
            elif current_task == "cvs_assessment" and "component_balanced_accuracy" in line.lower():
                try:
                    value = float(line.split(":")[-1].split("(")[0].strip())
                    metrics["cvs_acc"] = value
                except:
                    pass

    return metrics


def submit_model(file, model_name: str, organization: str, contact: str = "", model_url: str = "",
                 is_eccv: bool = False, team_name: str = "", authors: str = "",
                 eval_only: bool = False,
                 progress=gr.Progress()):
    """
    Process model submission: validate, evaluate, and (optionally) publish.

    The `contact` argument is the submitter's email (the form's "Email" field).
    For MedVidU ECCV 2026 Challenge participants (is_eccv=True), email, team
    name, and authors are mandatory; for general community submissions they
    are optional.

    When `eval_only=True`, the model is evaluated and a would-be ranking is
    returned, but no row is written to the public leaderboard and nothing is
    backed up to the private results repo. ECCV submissions cannot use
    eval-only — challenge entries must be published.

    Returns:
        (success, message)
    """
    # Normalize inputs
    model_name = (model_name or "").strip()
    organization = (organization or "").strip()
    contact = (contact or "").strip()
    team_name = (team_name or "").strip()
    authors = (authors or "").strip()
    eval_only = bool(eval_only) and not is_eccv  # ECCV can never be eval-only

    # Validation
    if not file:
        yield "❌ Please upload a results file"
        return

    if not model_name:
        yield "❌ Please provide a model name"
        return

    if not organization:
        yield "❌ Please provide an institution"
        return

    # ECCV-specific required fields
    if is_eccv:
        missing = []
        if not team_name:
            missing.append("Team Name")
        if not authors:
            missing.append("Authors")
        if not contact:
            missing.append("Email")
        if missing:
            yield f"❌ MedVidU ECCV 2026 Challenge submissions require: {', '.join(missing)}"
            return

    # Step 1: Check if model exists (only blocks publishing — eval-only can reuse names)
    progress(0.05, desc="Checking model name...")
    if eval_only:
        yield "🔍 **Step 1/6**: Skipping name uniqueness check (Evaluate Only)..."
    else:
        yield "🔍 **Step 1/6**: Checking if model name is available..."

    df = load_leaderboard()
    if not eval_only and model_name in df['model_name'].values:
        yield f"❌ Model '{model_name}' already exists in leaderboard. Please use a different name."
        return

    # Step 2: Validate file format
    progress(0.15, desc="Validating file format...")
    yield "📋 **Step 2/6**: Validating predictions file format..."

    valid, msg, has_precomputed_llm = validate_results_file(file.name)
    if not valid:
        yield f"❌ Invalid results file: {msg}"
        return

    yield f"✓ {msg}"

    # Step 3: Run evaluation with real-time log streaming
    progress(0.25, desc="Running evaluation...")
    import time

    # Start evaluation
    eval_wrapper = Path("evaluation/evaluate_predictions.py")
    output_dir = RESULTS_DIR / model_name.replace(" ", "_")
    output_dir.mkdir(exist_ok=True)
    input_file = output_dir / "input.json"
    shutil.copy(file.name, input_file)

    cmd = [
        sys.executable,
        "-u",  # Unbuffered output
        str(eval_wrapper),
        str(input_file),
        "--grouping", "overall",
        "--ground-truth", str(GROUND_TRUTH_FILE),
        "--skip-llm-judge"  # Skip DVC/VS/RC for faster testing
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}  # Force unbuffered
    )

    yield "⚙️ **Step 3/6**: Running evaluation (streaming logs)...\n\n```\nStarting evaluation subprocess...\n```"

    log_buffer = []
    last_update = time.time()
    last_heartbeat = time.time()
    line_count = 0
    start_time = time.time()

    # Non-blocking read with select (for timeout handling)
    import select

    while True:
        # Check if process has finished
        if process.poll() is not None:
            # Process finished, read any remaining output
            remaining = process.stdout.read()
            if remaining:
                for line in remaining.split('\n'):
                    line = line.rstrip()
                    if line.strip() and 'WARNING: All log messages' not in line:
                        log_buffer.append(line)
            break

        # Check for available output (with timeout)
        ready, _, _ = select.select([process.stdout], [], [], 0.5)

        if ready:
            # Read available line
            line = process.stdout.readline()
            if not line:
                break

            line = line.rstrip()
            if not line.strip():
                continue

            # Filter noise
            if 'WARNING: All log messages' in line:
                continue

            log_buffer.append(line)
            line_count += 1
            last_heartbeat = time.time()

        # Update UI every 0.5 seconds
        if time.time() - last_update > 0.5:
            # Show heartbeat if no logs yet
            if not log_buffer:
                elapsed = int(time.time() - start_time)
                log_text = f"⚙️ **Step 3/6**: Running evaluation...\n\n```\nWaiting for evaluation output... ({elapsed}s elapsed)\n```"
                yield log_text
            else:
                # Show last 25 lines
                recent = log_buffer[-25:]
                log_text = "⚙️ **Step 3/6**: Running evaluation...\n\n```\n"
                log_text += '\n'.join(recent)
                log_text += "\n```"
                yield log_text

            last_update = time.time()

            # Increment progress gradually from 25% to 75%
            # Assume ~500 lines of output, increment by 0.1% per line
            progress_increment = min(0.75, 0.25 + (line_count / 500) * 0.50)
            progress(progress_increment, desc="Running evaluation...")

    # Wait for process to fully complete
    process.wait()

    # Final log display
    if log_buffer:
        final_logs = log_buffer[-30:]
        log_text = "⚙️ **Step 3/6**: Evaluation completed\n\n```\n"
        log_text += '\n'.join(final_logs)
        log_text += "\n```"
        yield log_text

    # Save output
    with open(output_dir / "eval_output.txt", 'w') as f:
        f.write('\n'.join(log_buffer))

    # Check if evaluation succeeded
    if process.returncode != 0:
        yield f"\n❌ Evaluation failed (exit code {process.returncode})"
        return

    # Parse metrics from output
    full_output = '\n'.join(log_buffer)
    metrics = parse_evaluation_output(full_output)

    if not metrics:
        yield f"\n❌ Failed to parse evaluation metrics"
        return

    # Step 4: Check metrics
    progress(0.80, desc="Validating metrics...")
    yield "✓ Evaluation completed!"
    yield "🔍 **Step 4/6**: Validating extracted metrics..."

    # Fill missing metrics with 0 (allow partial submissions)
    caption_metrics = ['dvc_llm', 'dvc_f1', 'vs_llm', 'rc_llm']
    missing_metrics = [m for m in METRICS.keys() if m not in metrics]

    # Separate caption and other metrics
    missing_caption = [m for m in missing_metrics if m in caption_metrics]
    missing_other = [m for m in missing_metrics if m not in caption_metrics]

    # Fill all missing metrics with 0.0 (allow partial submissions)
    if missing_caption:
        for metric in missing_caption:
            metrics[metric] = 0.0
        yield f"⚠️  Skipped caption tasks, setting to 0: {missing_caption}"

    if missing_other:
        for metric in missing_other:
            metrics[metric] = 0.0
        yield f"⚠️  Missing tasks (setting to 0): {missing_other}"
        yield f"   Note: Partial submissions are allowed. Missing tasks will show as 0.0."

    computed_metrics = [m for m in METRICS.keys() if m in metrics and metrics[m] > 0]
    yield f"✓ Computed {len(computed_metrics)}/10 metrics (remaining set to 0.0)"

    # Step 5: Publish (or compute would-be rank in eval-only mode)
    clean_url = (model_url or "").strip()
    if clean_url and not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        clean_url = ""

    new_entry = {
        "model_name": model_name,
        "organization": organization,
        "team_name": team_name,
        "authors": authors,
        **{metric: round(metrics.get(metric, 0.0), 3) for metric in METRICS.keys()},
        "date": datetime.now().strftime("%Y-%m-%d"),
        "contact": contact,
        "model_url": clean_url,
        "is_eccv": bool(is_eccv),
        "status": "self_reported",
    }

    if eval_only:
        progress(0.90, desc="Computing would-be rank...")
        yield "📊 **Step 5/6**: Computing would-be rank (Evaluate Only — not publishing)..."
        # Rank against a temporary copy so the live leaderboard is untouched.
        preview_df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        preview_df = sort_by_avg_rank(preview_df)
        rank = preview_df[preview_df['model_name'] == model_name].index[0] + 1
        total = len(preview_df)
        yield "✓ Skipped publication — your run is NOT saved to the leaderboard or private repo."
    else:
        progress(0.90, desc="Adding to leaderboard...")
        yield "📊 **Step 5/6**: Adding model to leaderboard..."
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        # Sort by average rank across all metrics (lower avg rank = better)
        df = sort_by_avg_rank(df)
        save_leaderboard(df)
        yield "✓ Leaderboard updated!"
        # Backup results to private repo (non-blocking)
        backup_results_to_repo(model_name.replace(" ", "_"), output_dir)
        rank = df[df['model_name'] == model_name].index[0] + 1
        total = len(df)

    # Step 6: Build success message
    progress(1.0, desc="Complete!")
    if eval_only:
        yield "✅ **Step 6/6**: Evaluation complete (results not published)."
    else:
        yield "✅ **Step 6/6**: Submission complete!"

    # Build final success message
    if eval_only:
        success_msg = f"""
---

## 🧪 Evaluation Complete (Not Published)

This run was evaluated but **not added to the public leaderboard** and **not backed up** to the private results repo.
To publish, submit again with **"Evaluate Only"** unchecked.

**Model**: {model_name}
**Institution**: {organization}
"""
    else:
        success_msg = f"""
---

## ✅ Submission Successful!

**Model**: {model_name}
**Institution**: {organization}
"""
    if authors:
        success_msg += f"**Authors**: {authors}\n"
    if team_name:
        success_msg += f"**Team**: {team_name}\n"
    if is_eccv:
        success_msg += "**Track**: MedVidU ECCV 2026 Challenge\n"

    # Add note if LLM judge was skipped
    if has_precomputed_llm:
        success_msg += "\n📊 **Note**: Used pre-computed LLM judge scores from struc_info (skipped re-evaluation of DVC/VS/RC)\n"
    else:
        success_msg += "\n⚙️ **Note**: Full evaluation completed (including LLM judge for DVC/VS/RC)\n"

    success_msg += "\n### 📈 Metric Scores\n"
    for metric_key, metric_info in METRICS.items():
        score = metrics.get(metric_key, 0.0)
        success_msg += f"- **{metric_info['name']}**: {score:.3f}\n"

    if eval_only:
        success_msg += f"\n### 🏆 Would-Be Ranking\n**Rank**: #{rank} out of {total} models *(if published)*\n"
        success_msg += (
            "\n> **Heads up:** LLM-judge metrics (DVC_llm, VS_llm, RC_llm) are **0.0** in Evaluate Only mode "
            "because Step 2 (LLM Judge) requires a published row to read predictions from. "
            "Submit normally (Evaluate Only unchecked) when you want those scored.\n"
        )
        success_msg += "\nNothing has been saved. Submit again with Evaluate Only unchecked to publish."
    else:
        success_msg += f"\n### 🏆 Ranking\n**Rank**: #{rank} out of {total} models\n"
        success_msg += "\nRefresh the Leaderboard tab to see your model's position!"

    yield success_msg


def format_leaderboard_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format leaderboard dataframe for display with 10 metrics (no average).
    All metric values are rounded to 3 decimal places to match the project page table."""
    if df.empty:
        return df

    # Remove 'average' column if it exists (from old format)
    if 'average' in df.columns:
        df = df.drop('average', axis=1)

    # Create display dataframe with selected columns (no average)
    display_cols = ["rank", "model_name", "organization"]

    # Add metric columns in order
    for metric_key in METRICS.keys():
        if metric_key in df.columns:
            display_cols.append(metric_key)

    # Add date and contact
    display_cols.extend(["date", "contact"])

    # Filter to only existing columns
    display_cols = [col for col in display_cols if col in df.columns]

    # Rename columns for display
    display_df = df[display_cols].copy()

    # Round all metric columns to 3 decimal places for consistent display
    for metric_key in METRICS.keys():
        if metric_key in display_df.columns:
            display_df[metric_key] = display_df[metric_key].apply(
                lambda x: round(float(x), 3) if pd.notna(x) else 0.0
            )

    # Build column names
    column_names = []
    for col in display_cols:
        if col == "rank":
            column_names.append("Rank")
        elif col == "model_name":
            column_names.append("Model")
        elif col == "organization":
            column_names.append("Team")
        elif col == "date":
            column_names.append("Date")
        elif col == "contact":
            column_names.append("Contact")
        elif col in METRICS:
            column_names.append(METRICS[col]["name"])
        else:
            column_names.append(col)

    display_df.columns = column_names

    return display_df


def format_leaderboard_html(df: pd.DataFrame) -> str:
    """Render leaderboard as a styled HTML table with top-3 row highlighting.
    Colors match the MedGRPO project page: gold/silver/bronze for ranks 1-3."""
    if df.empty:
        return "<p style='text-align:center; color:#6b7280;'>No submissions yet.</p>"

    # Remove 'average' column if it exists
    if 'average' in df.columns:
        df = df.drop('average', axis=1)

    # Build column order
    metric_keys = [k for k in METRICS.keys() if k in df.columns]
    display_cols = ["rank", "model_name", "organization"] + metric_keys + ["date"]

    # Column headers
    header_map = {
        "rank": "Rank", "model_name": "Model", "organization": "Team", "date": "Date",
    }
    for k in metric_keys:
        header_map[k] = METRICS[k]["name"]

    # Row style per rank
    rank_styles = {
        1: "background:#fef2f2; font-weight:600;",   # gold/red tint
        2: "background:#fffbeb; font-weight:600;",   # silver/orange tint
        3: "background:#f0fdf4; font-weight:600;",   # bronze/green tint
    }
    rank_badges = {
        1: "🥇", 2: "🥈", 3: "🥉",
    }

    # Find best and 2nd-best per metric column (for cell highlighting)
    metric_best = {}
    metric_second = {}
    for k in metric_keys:
        vals = df[k].dropna().astype(float)
        if len(vals) >= 1:
            sorted_vals = vals.sort_values(ascending=False).unique()
            metric_best[k] = sorted_vals[0] if len(sorted_vals) >= 1 else None
            metric_second[k] = sorted_vals[1] if len(sorted_vals) >= 2 else None

    # Build HTML
    html = """<style>
.lb-table-wrap { background:#ffffff; color:#111827; border-radius:6px; }
.lb-table { width:100%; border-collapse:collapse; font-size:0.85rem; font-family:system-ui,-apple-system,sans-serif; background:#ffffff; color:#111827; }
.lb-table th { background:#f8fafc !important; color:#374151 !important; padding:8px 6px; text-align:center; border-bottom:2px solid #e5e7eb; font-size:0.78rem; position:sticky; top:0; z-index:1; }
.lb-table td { padding:6px; text-align:center; border-bottom:1px solid #f3f4f6; background:#ffffff; color:#111827; }
.lb-table td * { color:inherit !important; }
.lb-table td .eccv-badge { color:#ffd86b !important; }
.lb-table tr:hover td { background:#f9fafb; }
.lb-table tr[data-rank="1"] td { background:#fef2f2; font-weight:600; }
.lb-table tr[data-rank="2"] td { background:#fffbeb; font-weight:600; }
.lb-table tr[data-rank="3"] td { background:#f0fdf4; font-weight:600; }
.lb-table .model-col { text-align:left; font-weight:500; min-width:230px; white-space:nowrap; }
.lb-table .model-col a { color:#1e3a5f !important; }
.lb-table .org-col { text-align:left; color:#6b7280; font-size:0.8rem; white-space:nowrap; }
.lb-table .org-col * { color:#6b7280 !important; }
.lb-table .best-cell, .lb-table td.best-cell, .lb-table td.best-cell * { color:#b91c1c !important; font-weight:700; }
.lb-table .second-cell, .lb-table td.second-cell, .lb-table td.second-cell * { color:#b45309 !important; font-weight:600; }
@media (prefers-color-scheme: dark) {
  .lb-table-wrap { background:#111827; color:#e5e7eb; }
  .lb-table { background:#111827; color:#e5e7eb; }
  .lb-table th { background:#0f1e33 !important; color:#e5e7eb !important; border-bottom-color:#1f2937; }
  .lb-table td { background:#111827; color:#e5e7eb; border-bottom-color:#1f2937; }
  .lb-table tr:hover td { background:#1e293b; }
  .lb-table tr[data-rank="1"] td { background:#3b1f23; }
  .lb-table tr[data-rank="2"] td { background:#3b2e14; }
  .lb-table tr[data-rank="3"] td { background:#132b1d; }
  .lb-table .model-col a { color:#93c5fd !important; border-bottom-color:#93c5fd !important; }
  .lb-table .org-col, .lb-table .org-col * { color:#9ca3af !important; }
  .lb-table .best-cell, .lb-table td.best-cell, .lb-table td.best-cell * { color:#fca5a5 !important; }
  .lb-table .second-cell, .lb-table td.second-cell, .lb-table td.second-cell * { color:#fcd34d !important; }
}
</style>
<div class="lb-table-wrap" style="overflow-x:auto; max-height:600px; overflow-y:auto;">
<table class="lb-table">
<thead><tr>"""

    for col in display_cols:
        html += f"<th>{header_map.get(col, col)}</th>"
    html += "</tr></thead>\n<tbody>"

    for _, row in df.iterrows():
        rank = int(row.get('rank', 0))
        html += f'<tr data-rank="{rank}">'

        for col in display_cols:
            val = row.get(col, "")

            if col == "rank":
                badge = rank_badges.get(rank, "")
                html += f'<td style="white-space:nowrap;">{badge}{rank}</td>'

            elif col == "model_name":
                url = row.get("model_url", "")
                url = "" if (url is None or (isinstance(url, float) and pd.isna(url))) else str(url).strip()
                name_html = f'<a href="{url}" target="_blank" rel="noopener" style="color:#1e3a5f; text-decoration:none; border-bottom:1px dotted #1e3a5f;">{val}</a>' if url else f'{val}'
                is_eccv = row.get("is_eccv", False)
                if isinstance(is_eccv, float) and pd.isna(is_eccv):
                    is_eccv = False
                eccv_badge = (
                    ' <span class="eccv-badge" title="MedVidU @ ECCV 2026 Challenge submission" '
                    'style="display:inline-block; margin-left:6px; padding:1px 6px; '
                    'background:#1e3a5f; color:#ffd86b; font-size:0.68rem; font-weight:700; '
                    'border-radius:4px; vertical-align:middle;">ECCV&#39;26</span>'
                    if is_eccv else ""
                )
                html += f'<td class="model-col">{name_html}{eccv_badge}</td>'

            elif col == "organization":
                status = row.get("status", "")
                status_badge = {"api_verified": "&#9989; "}.get(status, "")
                html += f'<td class="org-col">{status_badge}{val}</td>'

            elif col == "date":
                html += f'<td style="color:#9ca3af; font-size:0.78rem;">{val}</td>'

            elif col in metric_keys:
                fval = round(float(val), 3) if pd.notna(val) else 0.0
                cell_class = ""
                if metric_best.get(col) is not None and fval == metric_best[col]:
                    cell_class = "best-cell"
                elif metric_second.get(col) is not None and fval == metric_second[col]:
                    cell_class = "second-cell"
                html += f'<td class="{cell_class}">{fval:.3f}</td>'

            else:
                html += f'<td>{val}</td>'

        html += "</tr>\n"

    html += "</tbody></table></div>"

    # Legend
    html += """<p style="text-align:center; margin-top:0.5rem; font-size:0.78rem; color:#6b7280;">
    <span style="color:#b91c1c; font-weight:700;">■ Best</span> &nbsp;
    <span style="color:#b45309; font-weight:600;">■ 2nd Best</span> &nbsp;&nbsp;
    🥇 1st &nbsp; 🥈 2nd &nbsp; 🥉 3rd overall &nbsp;&nbsp;
    &#9989; = User submission verified by maintainers via model API &nbsp;&nbsp;
    <span style="display:inline-block; padding:1px 6px; background:#1e3a5f; color:#ffd86b; font-size:0.68rem; font-weight:700; border-radius:4px;">ECCV&#39;26</span>
    = <a href="https://medvidu.github.io/" target="_blank" rel="noopener" style="color:#1e3a5f;">MedVidU @ ECCV 2026 Challenge</a> entry &nbsp;&nbsp;
    * = off-the-shelf models
    </p>"""

    return html


def check_needs_llm_judge(model_name: str) -> Tuple[bool, str]:
    """
    Check if a model needs LLM judge evaluation.

    Returns:
        (needs_llm_judge, message)
    """
    df = load_leaderboard()

    if model_name not in df['model_name'].values:
        return False, f"Model '{model_name}' not found"

    model_row = df[df['model_name'] == model_name].iloc[0]

    # Check if all three caption metrics are zero
    dvc_llm = model_row.get('dvc_llm', 0.0)
    vs_llm = model_row.get('vs_llm', 0.0)
    rc_llm = model_row.get('rc_llm', 0.0)

    if dvc_llm == 0.0 and vs_llm == 0.0 and rc_llm == 0.0:
        return True, "All caption metrics are 0.0, can run LLM judge"
    else:
        return False, "Caption metrics already computed"


def check_llm_judge_status(model_name: str) -> Tuple[str, str]:
    """
    Check the status of an ongoing LLM judge evaluation.

    Returns:
        (status, message)
        status: 'not_started', 'running', 'completed', 'failed'
    """
    model_dir = RESULTS_DIR / model_name.replace(" ", "_")
    status_file = model_dir / "llm_judge_status.json"

    if not status_file.exists():
        return 'not_started', 'No LLM judge evaluation in progress'

    try:
        with open(status_file, 'r') as f:
            status_data = json.load(f)

        status = status_data.get('status', 'not_started')
        progress = status_data.get('progress', '')
        timestamp = status_data.get('timestamp', '')

        if status == 'running':
            return 'running', f"Evaluation in progress: {progress}\nStarted: {timestamp}"
        elif status == 'completed':
            return 'completed', f"Evaluation completed: {timestamp}"
        elif status == 'failed':
            error = status_data.get('error', 'Unknown error')
            return 'failed', f"Evaluation failed: {error}"
        else:
            return 'not_started', 'No evaluation in progress'
    except Exception as e:
        return 'not_started', f"Error reading status: {e}"


def update_llm_judge_status(model_name: str, status: str, progress: str = "", error: str = ""):
    """Update the LLM judge evaluation status file."""
    model_dir = RESULTS_DIR / model_name.replace(" ", "_")
    status_file = model_dir / "llm_judge_status.json"

    status_data = {
        'status': status,
        'progress': progress,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if error:
        status_data['error'] = error

    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)


def run_llm_judge_evaluation(model_name: str, progress=gr.Progress()) -> str:
    """
    Start LLM judge evaluation in the background for DVC/VS/RC tasks.

    This function:
    1. Validates the model and checks if evaluation is needed
    2. Starts background evaluation process (can close browser)
    3. Returns immediately with status information

    Args:
        model_name: Name of the model to re-evaluate
        progress: Gradio progress tracker

    Returns:
        Status message (markdown)
    """
    try:
        # Check if model exists and needs LLM judge
        needs_llm, msg = check_needs_llm_judge(model_name)
        if not needs_llm:
            return f"❌ {msg}"

        # Check if evaluation is already running
        status, status_msg = check_llm_judge_status(model_name)
        if status == 'running':
            return f"⏳ **Evaluation Already Running**\n\n{status_msg}\n\nCheck status by refreshing or clicking 'Check Status' button."
        # Note: a stale 'completed' status with all-zero leaderboard scores
        # (needs_llm == True above) means the prior run produced no usable
        # metrics — likely API failures. Allow a rerun in that case.

        progress(0.1, desc="Validating...")
        yield f"🔍 **Validation**: Checking model predictions...\n\n"

        # Find the predictions file - try local first, then download from repo
        model_dir = RESULTS_DIR / model_name.replace(" ", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        input_file = model_dir / "input.json"

        if not input_file.exists():
            # Try to download from private HuggingFace repo
            try:
                token = os.environ.get('HF_TOKEN')
                if token:
                    yield f"⏳ Downloading predictions from private repository...\n\n"

                    from huggingface_hub import hf_hub_download

                    # Download the predictions file
                    predictions_path = hf_hub_download(
                        repo_id="UII-AI/MedVidBench-GroundTruth",
                        filename=f"results/{model_name.replace(' ', '_')}/input.json",
                        repo_type="dataset",
                        token=token,
                        cache_dir="./cache"
                    )

                    # Copy to local results directory
                    import shutil
                    shutil.copy(predictions_path, input_file)

                    yield f"✓ Downloaded predictions from repository\n\n"
                else:
                    yield f"❌ Predictions file not found locally and HF_TOKEN not available\n"
                    yield f"   Looked for: {input_file}\n"
                    return
            except Exception as e:
                yield f"❌ Predictions file not found: {input_file}\n"
                yield f"   Also failed to download from repository: {e}\n"
                return
        else:
            yield f"✓ Found predictions file locally\n\n"

        # Update status to running
        update_llm_judge_status(model_name, 'running', 'Starting evaluation...')

        # Start background process
        progress(0.2, desc="Starting background evaluation...")
        yield f"🚀 **Starting Background Evaluation**\n\n"
        yield f"⏳ This will take 10-20 minutes depending on API rate limits\n\n"
        yield f"✅ **You can close this browser tab** - evaluation runs in background\n\n"

        eval_wrapper = Path("evaluation/evaluate_predictions.py")
        log_file = model_dir / "eval_llm_judge_log.txt"

        # Build command for background execution.
        # Scope to caption tasks only — rule-based metrics were already computed
        # in Step 1 and are deterministic, so recomputing them here is wasted work.
        cmd = [
            sys.executable,
            "-u",
            str(eval_wrapper),
            str(input_file),
            "--grouping", "overall",
            "--ground-truth", str(GROUND_TRUTH_FILE),
            "--tasks", "dvc", "vs", "rc",
            # NOTE: No --skip-llm-judge flag, so LLM judge will run
        ]

        # Start process in background (detached)
        with open(log_file, 'w') as log_f:
            log_f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n")
            log_f.write("="*60 + "\n\n")

        # Launch background process that continues after app closes
        process = subprocess.Popen(
            cmd,
            stdout=open(log_file, 'a'),
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=True  # Detach from parent process
        )

        # Save PID for tracking
        pid_file = model_dir / "llm_judge_pid.txt"
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))

        progress(0.5, desc="Background process started...")

        success_msg = f"""
---

## ✅ Background Evaluation Started!

**Model**: {model_name}
**Process ID**: {process.pid}
**Started**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### ⏳ Evaluation Progress

The evaluation is now running in the background. This will take approximately 10-20 minutes.

### 📋 What's Happening

1. ⚙️ Running LLM judge on DVC/VS/RC tasks
2. 🔄 Using GPT-4 with retry logic (up to 5 attempts per sample)
3. 📊 Will automatically update leaderboard when complete

### ✅ You Can Now:

- ✓ **Close this browser tab** - evaluation continues running
- ✓ Come back later and check status using "Check Status" button
- ✓ Refresh the leaderboard in 10-20 minutes to see results

### 🔍 Check Status Later

1. Enter the same model name: `{model_name}`
2. Click "Check Status" button
3. Or refresh the leaderboard to see if metrics are updated

### 📝 Logs

Evaluation logs are being written to:
`{log_file}`
"""

        yield success_msg

        # Start background monitor thread to update status and leaderboard when complete
        import threading

        def monitor_and_update():
            """Monitor background process and update leaderboard when complete."""
            try:
                # Wait for process to complete
                process.wait()

                # Read final output
                with open(log_file, 'r') as f:
                    full_output = f.read()

                if process.returncode == 0:
                    # Parse metrics
                    metrics = parse_evaluation_output(full_output)

                    dvc_llm = metrics.get('dvc_llm', 0.0)
                    vs_llm = metrics.get('vs_llm', 0.0)
                    rc_llm = metrics.get('rc_llm', 0.0)

                    if dvc_llm > 0.0 or vs_llm > 0.0 or rc_llm > 0.0:
                        # Update leaderboard
                        df = load_leaderboard()
                        df.loc[df['model_name'] == model_name, 'dvc_llm'] = round(dvc_llm, 3)
                        df.loc[df['model_name'] == model_name, 'vs_llm'] = round(vs_llm, 3)
                        df.loc[df['model_name'] == model_name, 'rc_llm'] = round(rc_llm, 3)
                        df = sort_by_avg_rank(df)
                        save_leaderboard(df)

                        # Update status to completed
                        update_llm_judge_status(
                            model_name,
                            'completed',
                            f"DVC: {dvc_llm:.3f}, VS: {vs_llm:.3f}, RC: {rc_llm:.3f}"
                        )
                    else:
                        update_llm_judge_status(model_name, 'failed', error='Failed to extract metrics')
                else:
                    update_llm_judge_status(model_name, 'failed', error=f'Exit code {process.returncode}')

            except Exception as e:
                update_llm_judge_status(model_name, 'failed', error=str(e))

        # Start monitor thread (daemon so it doesn't block app shutdown)
        monitor_thread = threading.Thread(target=monitor_and_update, daemon=True)
        monitor_thread.start()

    except Exception as e:
        update_llm_judge_status(model_name, 'failed', error=str(e))
        yield f"❌ Error starting LLM judge evaluation: {str(e)}"


def check_llm_judge_evaluation_status(model_name: str) -> str:
    """Check and display status of LLM judge evaluation."""
    if not model_name or not model_name.strip():
        return "❌ Please enter a model name"

    status, msg = check_llm_judge_status(model_name.strip())

    if status == 'not_started':
        return f"ℹ️ **No Evaluation Running**\n\n{msg}"
    elif status == 'running':
        model_dir = RESULTS_DIR / model_name.replace(" ", "_")
        log_file = model_dir / "eval_llm_judge_log.txt"

        # Read last 30 lines of log
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                recent_lines = lines[-30:]

            log_preview = ''.join(recent_lines)

            return f"""
## ⏳ Evaluation Running

**Model**: {model_name}
**Status**: {msg}

### 📝 Recent Logs (last 30 lines)

```
{log_preview}
```

**Note**: Refresh this page or click "Check Status" again for updates.
"""
        except Exception as e:
            return f"⏳ **Evaluation Running**\n\n{msg}\n\n⚠️ Unable to read logs: {e}"

    elif status == 'completed':
        # Check if leaderboard was updated
        df = load_leaderboard()
        if model_name in df['model_name'].values:
            row = df[df['model_name'] == model_name].iloc[0]
            dvc = row.get('dvc_llm', 0.0)
            vs = row.get('vs_llm', 0.0)
            rc = row.get('rc_llm', 0.0)

            if dvc == 0.0 and vs == 0.0 and rc == 0.0:
                return f"""
## ⚠️ Evaluation Completed but All Scores Are 0.0

**Model**: {model_name}
**Completed**: {msg}

### 📈 Caption Metrics
- **DVC_llm**: {dvc:.3f}
- **VS_llm**: {vs:.3f}
- **RC_llm**: {rc:.3f}

The previous LLM judge run finished but produced no usable scores
(likely API failures or rate limiting). You can click **Run LLM Judge**
again to retry — the rerun is now allowed when all scores are 0.
"""

            return f"""
## ✅ Evaluation Complete!

**Model**: {model_name}
**Completed**: {msg}

### 📈 Caption Metrics
- **DVC_llm**: {dvc:.3f}
- **VS_llm**: {vs:.3f}
- **RC_llm**: {rc:.3f}

✓ Leaderboard has been updated!

Refresh the Leaderboard tab to see updated rankings.
"""
        else:
            return f"✓ **Evaluation Complete**\n\n{msg}\n\n⚠️ Model not found in leaderboard"

    elif status == 'failed':
        model_dir = RESULTS_DIR / model_name.replace(" ", "_")
        log_file = model_dir / "eval_llm_judge_log.txt"

        log_section = ""
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                log_preview = ''.join(lines[-50:]) if lines else "(log file is empty)"
                log_section = f"""
### 📝 Recent Logs (last 50 lines)

```
{log_preview}
```
"""
            else:
                log_section = f"\n⚠️ Log file not found: `{log_file}`\n"
        except Exception as e:
            log_section = f"\n⚠️ Unable to read logs: {e}\n"

        return f"""
## ❌ Evaluation Failed

**Model**: {model_name}
**Error**: {msg}
{log_section}
Please review the logs above and try running the evaluation again.
"""

    return f"ℹ️ **Status**: {status}\n\n{msg}"


# Create Gradio interface
with gr.Blocks(title="MedVidBench Leaderboard", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🏥 MedVidBench Leaderboard

    **MedVidBench** is a comprehensive benchmark for evaluating Video-Language Models on medical and surgical video understanding.
    It covers **8 tasks** across **8 surgical datasets** with **6,245 test samples**, evaluated on **10 metrics** including LLM-based caption judging.

    📄 [Paper](https://arxiv.org/abs/2512.06581) &nbsp; 🌐 [Project Page](https://uii-ai.github.io/MedGRPO/) &nbsp; 💾 [Dataset](https://huggingface.co/datasets/UII-AI/MedVidBench) &nbsp; 🤖 [Model](https://huggingface.co/UII-AI/uAI-NEXUS-MedVLM-1.0a-7B-RL) &nbsp; 💻 [GitHub](https://github.com/UII-AI/MedGRPO-Code) &nbsp; 🎮 [Demo](https://huggingface.co/spaces/UII-AI/MedGRPO-Demo)
    """)

    gr.HTML("""
    <div class="eccv-banner" style="
        background: linear-gradient(90deg, #1e3a5f 0%, #2e5a8f 100%) !important;
        padding: 14px 20px;
        border-radius: 8px;
        margin: 4px 0 16px 0;
        font-family: system-ui, -apple-system, sans-serif;
        box-shadow: 0 2px 6px rgba(30,58,95,0.15);
    ">
      <div style="font-size: 1.05rem; font-weight: 600; margin-bottom: 4px; color:#ffffff !important;">
        <span style="color:#ffffff !important;">🏅 Now hosting the </span>
        <a href="https://medvidu.github.io/" target="_blank" rel="noopener"
           style="color:#ffd86b !important; text-decoration:underline; font-weight:700;">MedVidU @ ECCV 2026 Challenge</a>
      </div>
      <div style="font-size: 0.9rem; color:#e6eef9 !important;">
        <span style="color:#e6eef9 !important;">
          Submissions are open to everyone. Challenge participants: tick the
          <b style="color:#ffffff !important;">&ldquo;MedVidU ECCV 2026 Challenge&rdquo;</b> box on the
          <b style="color:#ffffff !important;">📤 Submit Results</b> tab and provide your Team Name and Email.
        </span>
      </div>
    </div>
    """)

    with gr.Tabs():

        # Tab 1: Official Leaderboard (Verified)
        with gr.Tab("🏆 Official Leaderboard"):
            gr.Markdown("""
            ### Official Rankings (Verified)

            Models on this leaderboard have been **independently verified** by the benchmark maintainers.
            We evaluate top community submissions by requesting model API access and running our evaluation pipeline directly.

            This ensures **reproducible and trustworthy** results.
            """)

            def load_and_format_official_html():
                df = load_official_leaderboard()
                return format_official_leaderboard_html(df)

            official_table = gr.HTML(
                value=load_and_format_official_html(),
                label="Official Leaderboard"
            )

            with gr.Row():
                refresh_official_btn = gr.Button("🔄 Refresh", size="sm")
                official_status_text = gr.Markdown("", elem_classes="status-text")

            def refresh_official():
                df = load_official_leaderboard()
                num = len(df) if not df.empty else 0
                status = f"✓ {num} verified model(s)" if num > 0 else "No verified models yet"
                return format_official_leaderboard_html(df), status

            refresh_official_btn.click(
                fn=refresh_official,
                outputs=[official_table, official_status_text]
            )

            demo.load(
                fn=refresh_official,
                outputs=[official_table, official_status_text]
            )

            gr.Markdown("""
            ---

            #### How to get on the Official Leaderboard

            1. **Submit** your model predictions via the "Community Submissions" tab
            2. **Top performers** will be contacted by the benchmark maintainers
            3. **Provide model API access** so we can independently verify results
            4. Once verified, your model is added to the Official Leaderboard

            For questions, contact us via [GitHub](https://github.com/UII-AI/MedGRPO-Code).
            """)

        # Tab 2: Community Submissions
        with gr.Tab("📋 Community Submissions"):
            gr.Markdown("""
            ### Community Submissions

            Community members run inference on their own machines and upload predictions via the **📤 Submit Results** tab.
            Scores are then **evaluated on our server** against private ground truth.
            """)

            def load_and_format_leaderboard_html():
                """Load leaderboard and render as styled HTML table."""
                df = load_leaderboard()
                return format_leaderboard_html(df)

            def load_and_format_leaderboard():
                """Load and format leaderboard as DataFrame (used by admin/internal)."""
                df = load_leaderboard()
                if df.empty:
                    columns = ["rank", "model_name", "organization"] + list(METRICS.keys()) + ["date", "contact"]
                    return pd.DataFrame(columns=columns)
                return format_leaderboard_display(df)

            leaderboard_table = gr.HTML(
                value=load_and_format_leaderboard_html(),
                label="Leaderboard Rankings"
            )

            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Leaderboard", size="sm")
                status_text = gr.Markdown("", elem_classes="status-text")

            def refresh_leaderboard():
                """Refresh leaderboard and return status message."""
                df = load_leaderboard()
                num_models = len(df) if not df.empty else 0
                status = f"✓ Loaded {num_models} model(s)" if num_models > 0 else "No submissions yet"
                return format_leaderboard_html(df), status

            refresh_btn.click(
                fn=refresh_leaderboard,
                outputs=[leaderboard_table, status_text]
            )

            # Auto-load on page load
            demo.load(
                fn=refresh_leaderboard,
                outputs=[leaderboard_table, status_text]
            )

        # Tab 2: Submit
        with gr.Tab("📤 Submit Results"):
            gr.Markdown("""
            ### Submit Your Model Results

            Evaluation is a **two-step process**:

            | Step | What happens | Time |
            |------|-------------|------|
            | **Step 1** | Upload predictions -- evaluates CVS, NAP, SA, STG, TAG, DVC_F1 | ~2-5 min |
            | **Step 2** | Run LLM Judge -- evaluates DVC_llm, VS_llm, RC_llm caption quality | ~10-20 min (background) |

            ---

            ### Step 1: Upload Predictions

            Upload your model's **predictions only** on the **MedVidBench test set (6,245 samples)**.

            <details>
            <summary><b>Expected File Format</b> (click to expand)</summary>

            ```json
            [
              {
                "id": "video_id&&start&&end&&fps",
                "qa_type": "tal",
                "prediction": "Your model's answer here"
              },
              {
                "id": "another_video&&0&&10&&1.0",
                "qa_type": "video_summary",
                "prediction": "The surgeon performs..."
              }
            ]
            ```

            **Required fields**:
            - `id`: Sample identifier (matches test data from HuggingFace dataset)
            - `qa_type`: Task type (tal/stg/next_action/dense_captioning/video_summary/region_caption/skill_assessment/cvs_assessment)
            - `prediction`: Your model's answer (text output)

            </details>

            **Important**: Submit **predictions only** (no ground truth needed). The server merges with private ground truth and evaluates securely.
            """)

            with gr.Row():
                with gr.Column():
                    eccv_checkbox = gr.Checkbox(
                        label="I am submitting to the MedVidU ECCV 2026 Challenge",
                        value=False,
                        info="Tick this to reveal challenge fields. Team Name, Authors, and Email become required."
                    )

                    eval_only_checkbox = gr.Checkbox(
                        label="Evaluate Only (don't publish to the leaderboard)",
                        value=False,
                        info="Run evaluation and see your would-be rank without adding a row to the public table. LLM-judge metrics (DVC_llm, VS_llm, RC_llm) cannot be computed in this mode — they require a published row. Disabled for ECCV submissions."
                    )

                    model_name_input = gr.Textbox(
                        label="Model Name",
                        placeholder="e.g., uAI-NEXUS-MedVLM-1.0a-7B-RL",
                        info="Unique identifier for your model"
                    )

                    org_input = gr.Textbox(
                        label="Institution",
                        placeholder="e.g., University Name",
                        info="Organization or institution name"
                    )

                    email_input = gr.Textbox(
                        label="Email",
                        placeholder="email@example.com",
                        info="Optional for community submissions; required for ECCV 2026 Challenge."
                    )

                    model_url_input = gr.Textbox(
                        label="Model URL (Optional)",
                        placeholder="https://huggingface.co/your-org/your-model",
                        info="Link to model card, paper, or project page — your model name becomes a clickable link"
                    )

                    # ── ECCV Challenge-only fields (hidden until checkbox is ticked) ──
                    team_name_input = gr.Textbox(
                        label="Team Name",
                        placeholder="e.g., Team MedVision",
                        info="Required for MedVidU ECCV 2026 Challenge participants",
                        visible=False,
                    )

                    authors_input = gr.Textbox(
                        label="Authors",
                        placeholder="e.g., Jane Doe, John Smith",
                        info="Comma-separated list of contributors. Required for ECCV 2026 Challenge.",
                        visible=False,
                    )

                with gr.Column():
                    results_file_input = gr.File(
                        label="Upload Results JSON",
                        file_types=[".json"],
                        file_count="single"
                    )

            # Show/hide ECCV-only fields when checkbox toggles, and disable
            # Evaluate Only (ECCV submissions must be published).
            def _toggle_eccv_fields(is_eccv):
                vis = gr.update(visible=bool(is_eccv))
                eval_update = (
                    gr.update(value=False, interactive=False) if is_eccv
                    else gr.update(interactive=True)
                )
                return vis, vis, eval_update

            eccv_checkbox.change(
                fn=_toggle_eccv_fields,
                inputs=[eccv_checkbox],
                outputs=[team_name_input, authors_input, eval_only_checkbox],
            )

            submit_btn = gr.Button("🚀 Step 1: Submit & Evaluate", variant="primary", size="lg")

            submission_output = gr.Markdown(label="Submission Status")

            # Wire up submission with progress tracking
            submit_btn.click(
                fn=submit_model,
                inputs=[
                    results_file_input, model_name_input, org_input, email_input, model_url_input,
                    eccv_checkbox, team_name_input, authors_input, eval_only_checkbox,
                ],
                outputs=submission_output
            )

            # ── Step 2: LLM Judge ──
            gr.Markdown("""
            ---

            ### Step 2: Run LLM Judge (Caption Metrics)

            After Step 1 completes, the caption metrics (DVC_llm, VS_llm, RC_llm) will be **0.0**.
            Run the LLM Judge here to compute them using GPT-4.1/Gemini.

            - Enter the **exact model name** you used in Step 1
            - The evaluation runs **in the background** -- you can close the browser and come back later
            - Check progress anytime with the "Check Status" button
            """)

            with gr.Row():
                llm_judge_model_input = gr.Textbox(
                    label="Model Name",
                    placeholder="Enter exact model name from Step 1",
                    scale=3
                )
                with gr.Column(scale=1):
                    run_llm_judge_btn = gr.Button("🚀 Step 2: Run LLM Judge", variant="primary")
                    check_status_btn = gr.Button("🔍 Check Status", variant="secondary")

            llm_judge_output = gr.Markdown(label="LLM Judge Status")

            # Wire up LLM judge evaluation
            run_llm_judge_btn.click(
                fn=run_llm_judge_evaluation,
                inputs=[llm_judge_model_input],
                outputs=llm_judge_output
            )

            # Wire up status check
            check_status_btn.click(
                fn=check_llm_judge_evaluation_status,
                inputs=[llm_judge_model_input],
                outputs=llm_judge_output
            )

        # Tab 3: About (merged Tasks & Metrics + About)
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ### About MedVidBench

            **MedVidBench** is a comprehensive benchmark for evaluating Video-Language Models on medical and surgical video understanding,
            introduced in the **MedGRPO** paper. It spans **8 tasks** across **8 surgical datasets** with **6,245 test samples**.

            ---

            ### How Models Are Ranked

            Models are ranked by **average rank across all 10 metrics** — lower average rank = better. For each metric we rank every distinct score (1 = best; ties share a rank and the next distinct score gets the next consecutive rank), then average those per-metric ranks. This is robust to different metric scales (accuracy 0–1 vs. LLM-judge 1–5), avoids artificial gaps caused by tied rounded scores, and rewards models that are strong across tasks rather than exceptional on one.

            **Global ranking across views:** the rank shown is computed against the **union of all submissions** (official ∪ community), so the same model gets the same rank number in either the Official or the Community table — even though each table only displays a subset of rows. The Official table omits rows from the global ranking; the rank column shows each row's position in the full ranking, not its position within the visible subset.

            **Tiebreakers** (applied in order when two models have the same average rank):
            1. **Number of metrics won outright** — a model that's #1 on more metrics wins over one that ties closely on many.
            2. **Sum of per-metric ranks** — catches near-ties where the mean rounded equal.
            3. **Sum of normalized scores** — favors the model with marginally higher absolute scores.
            4. **Model name alphabetical** — final fallback for full determinism.

            ---

            ### Benchmark Tasks
            """)

            # Create tasks table
            tasks_data = []
            for task_key, task_info in TASKS.items():
                tasks_data.append({
                    "Task": task_info["name"],
                    "Key": task_info["key"],
                    "Metrics": task_info["metrics"],
                    "Description": task_info["description"]
                })

            tasks_df = pd.DataFrame(tasks_data)
            gr.Dataframe(value=tasks_df, interactive=False)

            gr.Markdown("""
            ### Evaluation Metrics

            - **CVS Assessment** (`CVS_acc`): Accuracy on clinical variable scoring (Cholec80_CVS)
            - **Next Action Prediction** (`NAP_acc`): Classification accuracy for the next surgical step
            - **Skill Assessment** (`SA_acc`): Surgical skill level classification accuracy (JIGSAWS)
            - **Spatiotemporal Grounding** (`STG_mIoU`): Mean IoU over the joint spatial + temporal region
            - **Temporal Action Grounding** (`TAG_mIoU@0.3`, `TAG_mIoU@0.5`): Mean IoU over temporal segments, computed at two IoU thresholds (0.3 and 0.5)
            - **Dense Video Captioning** (`DVC_F1`, `DVC_llm`): F1 over predicted vs. ground-truth temporal windows, plus LLM-judge caption quality
            - **Video Summary** (`VS_llm`): LLM-judge caption quality scoring
            - **Region Caption** (`RC_llm`): LLM-judge caption quality scoring

            #### LLM Judge Details

            Caption tasks (DVC, VS, RC) use GPT-4.1 or Gemini-Pro with rubric-based scoring (1-5 scale) across 5 key aspects:
            **R2** (Relevance & Medical Terminology), **R4** (Actionable Surgical Actions), **R5** (Comprehensive Detail Level),
            **R7** (Anatomical & Instrument Precision), **R8** (Clinical Context & Coherence).
            The final score is the average across these 5 aspects.

            ---

            ### Test Set Statistics

            - **Total samples**: 6,245
            - **Source datasets**: 8 (AVOS, CholecT50, CholecTrack20, Cholec80_CVS, CoPESD, EgoSurgery, NurViD, JIGSAWS)
            - **Video frames**: ~103,742
            - **Training samples**: 51,505

            ---

            ### Citation

            If you use our model or benchmark (MedVidBench / uAI-NEXUS-MedVLM), please cite our paper. To ensure reproducibility and acknowledge the significant investment in establishing this benchmark, please use the following official citation in any published work or public repository:

            ```bibtex
            @inproceedings{su2026medgrpo,
              title     = {{MedGRPO}: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding},
              author    = {Su, Yuhao and Choudhuri, Anwesa and Gao, Zhongpai and Planche, Benjamin and
                           Nguyen, Van Nguyen and Zheng, Meng and Shen, Yuhan and Innanje, Arun and
                           Chen, Terrence and Elhamifar, Ehsan and Wu, Ziyan},
              booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
              year      = {2026}
            }
            ```

            ### License

            - **Dataset**: CC BY-NC-SA 4.0 (Non-commercial, Share-alike)
            - **Leaderboard Code**: Apache 2.0
            - **Evaluation Scripts**: MIT

            ### Contact

            For questions or issues, open an issue on [GitHub](https://github.com/UII-AI/MedGRPO-Code) or visit the [project page](https://uii-ai.github.io/MedGRPO/).
            """)

        # Tab 5: Admin Panel (Password Protected)
        with gr.Tab("🔒 Admin"):
            gr.Markdown("""
            ### Admin Panel

            Manage both the **Official Leaderboard** (verified models) and **Community Submissions**.

            **Note**: Admin password is set via `ADMIN_PASSWORD` environment variable in HuggingFace Spaces settings.
            """)

            # Password authentication
            with gr.Row():
                admin_password_input = gr.Textbox(
                    label="Admin Password",
                    type="password",
                    placeholder="Enter admin password",
                    scale=3
                )
                login_btn = gr.Button("🔓 Login", variant="primary", scale=1)

            login_status = gr.Markdown("", visible=True)

            # Admin panel (hidden by default, shown after successful login)
            with gr.Column(visible=False) as admin_panel:

                # ── Official Leaderboard Management ──
                gr.Markdown("---\n### ✅ Official Leaderboard Management")
                gr.Markdown("Manage verified models. Promote from community submissions or add manually.")

                # Helper to build admin view of official leaderboard
                def get_official_for_admin():
                    df = load_official_leaderboard()
                    if df.empty:
                        return pd.DataFrame(columns=["rank", "model_name", "organization", "verified_date", "contact"])
                    cols = ["rank", "model_name", "organization", "verified_date", "contact"]
                    available = [c for c in cols if c in df.columns]
                    return df[available]

                official_admin_table = gr.Dataframe(
                    value=get_official_for_admin(),
                    interactive=False,
                    label="Official Leaderboard Entries",
                    wrap=True
                )

                gr.Markdown("#### Promote from Community Submissions")
                gr.Markdown("Copy an exact model name from the community table below to promote it to the official leaderboard.")
                with gr.Row():
                    promote_model_input = gr.Textbox(
                        label="Model Name to Promote",
                        placeholder="Exact model name from community submissions",
                        scale=3
                    )
                    promote_btn = gr.Button("⬆️ Promote to Official", variant="primary", scale=1)

                promote_status = gr.Markdown("")

                gr.Markdown("#### Add Model Manually")
                gr.Markdown("Add a model directly with scores you've verified via API access.")
                with gr.Row():
                    manual_model_name = gr.Textbox(label="Model Name", placeholder="e.g., Qwen2.5-VL-7B", scale=2)
                    manual_org = gr.Textbox(label="Organization", placeholder="e.g., University Name", scale=2)
                    manual_contact = gr.Textbox(label="Contact (optional)", placeholder="email", scale=1)

                with gr.Row():
                    manual_cvs_acc = gr.Number(label="CVS_acc", value=0.0, precision=3)
                    manual_nap_acc = gr.Number(label="NAP_acc", value=0.0, precision=3)
                    manual_sa_acc = gr.Number(label="SA_acc", value=0.0, precision=3)
                    manual_stg_miou = gr.Number(label="STG_mIoU", value=0.0, precision=3)
                    manual_tag_03 = gr.Number(label="TAG_mIoU@0.3", value=0.0, precision=3)

                with gr.Row():
                    manual_tag_05 = gr.Number(label="TAG_mIoU@0.5", value=0.0, precision=3)
                    manual_dvc_f1 = gr.Number(label="DVC_F1", value=0.0, precision=3)
                    manual_dvc_llm = gr.Number(label="DVC_llm", value=0.0, precision=3)
                    manual_vs_llm = gr.Number(label="VS_llm", value=0.0, precision=3)
                    manual_rc_llm = gr.Number(label="RC_llm", value=0.0, precision=3)

                add_manual_btn = gr.Button("➕ Add to Official Leaderboard", variant="primary")
                manual_status = gr.Markdown("")

                gr.Markdown("#### Remove from Official Leaderboard")
                with gr.Row():
                    remove_official_input = gr.Textbox(
                        label="Model Name to Remove",
                        placeholder="Exact model name",
                        scale=3
                    )
                    remove_official_btn = gr.Button("🗑️ Remove from Official", variant="stop", scale=1)

                remove_official_status = gr.Markdown("")

                refresh_official_admin_btn = gr.Button("🔄 Refresh Official Table", size="sm")

                # ── Community Submissions Management ──
                gr.Markdown("---\n### 📋 Community Submissions Management")

                admin_table = gr.Dataframe(
                    value=get_leaderboard_for_admin(),
                    interactive=False,
                    label="Community Leaderboard Entries",
                    wrap=True
                )

                with gr.Row():
                    refresh_admin_btn = gr.Button("🔄 Refresh List", size="sm")
                    delete_model_input = gr.Textbox(
                        label="Model Name to Delete",
                        placeholder="Enter exact model name",
                        scale=2
                    )
                    delete_btn = gr.Button("🗑️ Delete Model", variant="stop", scale=1)

                delete_status = gr.Markdown("")

                gr.Markdown("""
                ---

                ### 🔐 Security Notes

                - Set `ADMIN_PASSWORD` in HuggingFace Spaces Settings Secrets
                - Admin actions affect both community and official leaderboards
                - **Deletion is permanent and cannot be undone!**
                """)

            # Login handler
            def handle_login(password):
                if check_admin_password(password):
                    return (
                        "✓ Login successful! Admin panel unlocked.",
                        gr.update(visible=True),
                        get_official_for_admin(),
                        get_leaderboard_for_admin(),
                    )
                else:
                    return (
                        "❌ Invalid password. Please try again.",
                        gr.update(visible=False),
                        get_official_for_admin(),
                        get_leaderboard_for_admin(),
                    )

            login_btn.click(
                fn=handle_login,
                inputs=[admin_password_input],
                outputs=[login_status, admin_panel, official_admin_table, admin_table]
            )

            # ── Official Leaderboard handlers ──

            def handle_promote(model_name):
                if not model_name or not model_name.strip():
                    return "❌ Please enter a model name", get_official_for_admin()
                success, msg = promote_to_official(model_name.strip())
                prefix = "## ✓ Promoted\n\n" if success else "## ❌ Failed\n\n"
                return prefix + msg, get_official_for_admin()

            promote_btn.click(
                fn=handle_promote,
                inputs=[promote_model_input],
                outputs=[promote_status, official_admin_table]
            )

            def handle_manual_add(name, org, contact, cvs, nap, sa, stg, t03, t05, df1, dllm, vllm, rllm):
                if not name or not name.strip():
                    return "❌ Please enter a model name", get_official_for_admin()
                if not org or not org.strip():
                    return "❌ Please enter an organization", get_official_for_admin()
                metrics = {
                    "cvs_acc": cvs, "nap_acc": nap, "sa_acc": sa, "stg_miou": stg,
                    "tag_miou_03": t03, "tag_miou_05": t05,
                    "dvc_f1": df1, "dvc_llm": dllm, "vs_llm": vllm, "rc_llm": rllm,
                }
                success, msg = add_to_official_leaderboard(name.strip(), org.strip(), metrics, contact.strip() if contact else "")
                prefix = "## ✓ Added\n\n" if success else "## ❌ Failed\n\n"
                return prefix + msg, get_official_for_admin()

            add_manual_btn.click(
                fn=handle_manual_add,
                inputs=[manual_model_name, manual_org, manual_contact,
                        manual_cvs_acc, manual_nap_acc, manual_sa_acc, manual_stg_miou,
                        manual_tag_03, manual_tag_05, manual_dvc_f1, manual_dvc_llm,
                        manual_vs_llm, manual_rc_llm],
                outputs=[manual_status, official_admin_table]
            )

            def handle_remove_official(model_name):
                if not model_name or not model_name.strip():
                    return "❌ Please enter a model name", get_official_for_admin()
                success, msg = remove_from_official_leaderboard(model_name.strip())
                prefix = "## ✓ Removed\n\n" if success else "## ❌ Failed\n\n"
                return prefix + msg, get_official_for_admin()

            remove_official_btn.click(
                fn=handle_remove_official,
                inputs=[remove_official_input],
                outputs=[remove_official_status, official_admin_table]
            )

            refresh_official_admin_btn.click(
                fn=get_official_for_admin,
                outputs=[official_admin_table]
            )

            # ── Community Submissions handlers ──

            def refresh_admin_table():
                return get_leaderboard_for_admin()

            refresh_admin_btn.click(
                fn=refresh_admin_table,
                outputs=[admin_table]
            )

            def handle_delete(model_name):
                if not model_name or not model_name.strip():
                    return "❌ Please enter a model name", get_leaderboard_for_admin()

                success, message = delete_model_submission(model_name.strip())

                if success:
                    return f"## ✓ Deletion Successful\n\n{message}", get_leaderboard_for_admin()
                else:
                    return f"## ❌ Deletion Failed\n\n{message}", get_leaderboard_for_admin()

            delete_btn.click(
                fn=handle_delete,
                inputs=[delete_model_input],
                outputs=[delete_status, admin_table]
            )

if __name__ == "__main__":
    # Launch with queue for better concurrency
    demo.queue(default_concurrency_limit=5)
    demo.launch(
        share=True,
        server_name="0.0.0.0"
    )
