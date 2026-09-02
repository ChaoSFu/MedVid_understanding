# MedVidBench Leaderboard Evaluation

Auto-detection wrapper for evaluating predictions with automatic ground-truth merging.

## Overview

This evaluation system supports two input formats:

1. **Merged format** (already contains ground-truth): Like `results.json`
2. **Prediction-only format** (needs ground-truth): Like user submissions

The wrapper automatically detects which format you're using and handles ground-truth merging if needed.

## Quick Start

```bash
# Evaluate predictions (auto-detects format)
python evaluate_predictions.py <predictions_file>

# Evaluate specific tasks
python evaluate_predictions.py <predictions_file> --tasks tal stg

# Only analyze file structure
python evaluate_predictions.py <predictions_file> --analyze-only

# Use overall grouping (aggregate all datasets)
python evaluate_predictions.py <predictions_file> --grouping overall
```

## Input Formats

### Format 1: Prediction-Only (User Submission Format)

```json
[
  {
    "id": "kcOqlifSukA&&22425&&25124&&1.0",
    "qa_type": "tal",
    "prediction": "22.0-78.0, 89.0-94.0 seconds."
  },
  ...
]
```

**ID Format**: `video_id&&start_frame&&end_frame&&fps`

**Required fields**:
- `id`: Unique identifier matching ground-truth
- `qa_type`: Task type (tal, stg, dvc, next_action, rc, vs, skill_assessment, cvs_assessment)
- `prediction`: Model's prediction text

**What happens**: The wrapper automatically merges with `ground_truth.json` to create complete evaluation records.

### Format 2: Merged (Complete Format)

```json
{
  "0": {
    "metadata": {
      "video_id": "kcOqlifSukA",
      "fps": "1.0",
      "input_video_start_frame": "22425",
      "input_video_end_frame": "25124"
    },
    "qa_type": "tal",
    "struc_info": [...],
    "question": "...",
    "gnd": "0.0-10.0 seconds.",
    "answer": "22.0-78.0, 89.0-94.0 seconds.",
    "data_source": "AVOS"
  },
  ...
}
```

**Required fields**:
- `metadata`: Video metadata (video_id, fps, frame range)
- `qa_type`: Task type
- `struc_info`: Ground-truth structured information
- `question`: Question text
- `gnd`: Ground-truth answer
- `answer`: Model prediction
- `data_source`: Dataset name

**What happens**: The wrapper uses the file directly for evaluation.

## Ground-Truth File

**Location**: `/root/code/MedVidBench-Leaderboard/data/ground_truth.json`

**Structure**: Array of records, each containing:
- Complete metadata (video_id, fps, frame range)
- `struc_info`: Structured ground-truth (spans for TAL/STG, boxes for RC, etc.)
- Ground-truth answer
- Dataset source

**Note**: This file is NOT public. Users submit prediction-only files, which are merged server-side.

## Supported Tasks

| Task | qa_type | Metrics |
|------|---------|---------|
| **TAL** | `tal` | Recall@0.3/0.5, mIoU@0.3/0.5 |
| **STG** | `stg` | IoU@0.3/0.5/0.7, mIoU |
| **DVC** | `dense_captioning`, `dense_captioning_gpt`, `dense_captioning_gemini`, `dc` | CIDER, METEOR, Precision, Recall, F1, SODA_c |
| **Next Action** | `next_action` | Accuracy (per-dataset) |
| **RC** | `region_caption`, `region_caption_gpt`, `region_caption_gemini` | LLM Judge (GPT-4.1/Gemini) |
| **VS** | `video_summary`, `video_summary_gpt`, `video_summary_gemini` | LLM Judge (GPT-4.1/Gemini) |
| **Skill Assessment** | `skill_assessment` | Accuracy, Macro F1, Weighted F1 |
| **CVS Assessment** | `cvs_assessment` | Accuracy, Precision, Recall, F1 |

## Usage Examples

### Example 1: Evaluate User Submission (Prediction-Only)

```bash
# User submits predictions in prediction-only format
python evaluate_predictions.py user_predictions.json

# Output:
# [EvaluationWrapper] ✓ Detected: Prediction-only format (id, qa_type, prediction)
# [EvaluationWrapper] Merging with ground-truth...
# [EvaluationWrapper] ✓ Successfully merged 6245/6245 predictions
# ... [evaluation results] ...
```

### Example 2: Evaluate Internal Results (Already Merged)

```bash
# Internal evaluation with complete data
python evaluate_predictions.py results.json

# Output:
# [EvaluationWrapper] ✓ Detected: Predictions already contain ground-truth
# [EvaluationWrapper] Using predictions file directly for evaluation
# ... [evaluation results] ...
```

### Example 3: Specific Tasks Only

```bash
# Evaluate only TAL and STG tasks
python evaluate_predictions.py predictions.json --tasks tal stg

# Evaluate captioning tasks with LLM judge
python evaluate_predictions.py predictions.json --tasks rc vs dvc
```

### Example 4: Different Grouping Modes

```bash
# Per-dataset results (default)
python evaluate_predictions.py predictions.json --grouping per-dataset

# Overall results (aggregate all datasets)
python evaluate_predictions.py predictions.json --grouping overall
```

### Example 5: Skip LLM Judge (Use Pre-computed Scores)

```bash
# Skip LLM judge evaluation for caption tasks
# Useful when LLM scores are already pre-computed in the predictions
python evaluate_predictions.py predictions.json --skip-llm-judge
```

### Example 6: Analyze File Structure

```bash
# Only analyze what tasks/datasets are present
python evaluate_predictions.py predictions.json --analyze-only

# Output:
# Found QA types:
#   tal: 1637 records
#   stg: 780 records
#   ...
# Found datasets:
#   AVOS: 321 records
#   CholecT50: 409 records
#   ...
```

## Command-Line Options

```
python evaluate_predictions.py PREDICTIONS_FILE [OPTIONS]

Required:
  PREDICTIONS_FILE          Path to predictions JSON (merged or prediction-only format)

Optional:
  --ground-truth PATH       Path to ground-truth JSON (default: data/ground_truth.json)
  --tasks TASK [TASK ...]  Specific tasks to evaluate (default: all available)
                           Choices: dvc, tal, next_action, stg, rc, vs,
                                   skill_assessment, cvs_assessment
  --grouping {per-dataset,overall}
                           Grouping strategy (default: per-dataset)
                           - per-dataset: Results per dataset
                           - overall: Aggregate all datasets
  --analyze-only           Only analyze file structure, no evaluation
  --skip-llm-judge         Skip LLM judge for caption tasks (use pre-computed scores)
  -h, --help              Show help message
```

## Output Format

### Per-Dataset Grouping (Default)

```
================================================================================
EVALUATION RESULTS - PER DATASET
================================================================================

AVOS:
  TAL:
    recall@0.3: 0.45
    meanIoU@0.3: 0.42
    recall@0.5: 0.32
    meanIoU@0.5: 0.28

CholecT50:
  TAL:
    recall@0.3: 0.52
    ...
```

### Overall Grouping

```
================================================================================
EVALUATION RESULTS - OVERALL (Dataset-Agnostic)
================================================================================

TAL - Overall Evaluation (All Datasets Combined)
Total samples: 1637

  recall@0.3: 0.48
  meanIoU@0.3: 0.45
  recall@0.5: 0.35
  meanIoU@0.5: 0.30
```

## Workflow for User Submissions

1. **User downloads benchmark**: `/root/code/MedVidBench/cleaned_test_data_11_04.json`
   - Contains questions but NO ground-truth (struc_info removed)

2. **User runs inference**: Generates predictions for each sample

3. **User submits predictions**: prediction-only format
   ```json
   [
     {
       "id": "<from benchmark>",
       "qa_type": "<from benchmark>",
       "prediction": "<model output>"
     },
     ...
   ]
   ```

4. **Server evaluates**:
   ```bash
   python evaluate_predictions.py user_submission.json
   ```
   - Auto-detects format
   - Merges with server-side ground-truth
   - Runs evaluation
   - Returns metrics

## File Structure

```
evaluation/
├── README.md                        # This file
├── evaluate_predictions.py          # Main wrapper (auto-detection + merging)
├── evaluate_all_pai.py             # Core evaluation orchestrator
├── eval_tal.py                     # TAL evaluation
├── eval_stg.py                     # STG evaluation
├── eval_dvc.py                     # Dense captioning evaluation
├── eval_next_action.py             # Next action evaluation
├── eval_caption_llm_judge.py       # RC/VS LLM judge evaluation
├── eval_skill_assessment.py        # Skill assessment evaluation
└── eval_cvs_assessment.py          # CVS assessment evaluation
```

## Key Features

### Auto-Detection Logic

The wrapper detects format by checking for these indicators:

**Prediction-only format**:
- Has `id` field (video_id&&start&&end&&fps)
- Has `prediction` field
- Missing `gnd` or `struc_info`

**Merged format**:
- Has `question` field
- Has `gnd` field (ground-truth)
- Has `struc_info` field (structured GT)
- Has `metadata` dict

### Ground-Truth Merging

When prediction-only format is detected:

1. Load predictions and ground-truth
2. Build index: `{id -> ground_truth_record}`
3. For each prediction:
   - Look up ground-truth by ID
   - Merge into complete record
   - Add `data_source` from ground-truth
4. Save to temporary file
5. Run evaluation
6. Clean up temporary file

### Dataset Detection

Datasets are detected from:
1. **`data_source` field** (primary, leaderboard format)
2. `dataset` field (fallback)
3. `dataset_name` field (fallback)
4. Video ID patterns (last resort):
   - YouTube IDs (11 chars with letters) → AVOS
   - `*_part*` pattern → CoPESD
   - `video*` pattern → CholecT50

## Error Handling

### Missing Ground-Truth

```bash
# If ground-truth file not found
[EvaluationWrapper] ❌ ERROR: Ground-truth file not found: /path/to/ground_truth.json
```

**Solution**: Specify correct path with `--ground-truth`

### Unmatched Predictions

```bash
[EvaluationWrapper] ⚠️  WARNING: 10 predictions not found in ground-truth
[EvaluationWrapper]   First 5 unmatched IDs: [...]
```

**Cause**: Prediction IDs don't match ground-truth IDs

**Solution**: Check ID format (video_id&&start&&end&&fps must match exactly)

### Invalid ID Format

```bash
ValueError: Invalid ID format: <id_string>
```

**Cause**: ID doesn't follow `video_id&&start&&end&&fps` format

**Solution**: Fix ID format in predictions

## API Keys for LLM Judge

For RC/VS evaluation with LLM judge:

```bash
export OPENAI_API_KEY="your-key"      # For GPT-4.1
export GOOGLE_API_KEY="your-key"      # For Gemini

# Then run evaluation
python evaluate_predictions.py predictions.json --tasks rc vs
```

**Cost**: ~$0.012 per RC/VS sample (GPT-4.1)

## Verification Checklist

Before evaluating submissions:

```bash
# 1. Check file format
python evaluate_predictions.py submission.json --analyze-only

# 2. Verify ground-truth file exists
ls -lh /root/code/MedVidBench-Leaderboard/data/ground_truth.json

# 3. Run evaluation on sample (first 100 records)
head -100 submission.json > sample.json
python evaluate_predictions.py sample.json

# 4. If successful, run full evaluation
python evaluate_predictions.py submission.json
```

## Performance

- **Small files** (100 samples): ~5-10 seconds
- **Full benchmark** (6245 samples): ~2-5 minutes
  - TAL/STG: ~30 seconds per dataset
  - Next Action: ~20 seconds per dataset
  - DVC: ~1-2 minutes (metric computation)
  - RC/VS with LLM judge: ~5-10 minutes (API calls)

## Notes

- Ground-truth file contains **1414 records** (subset for leaderboard testing)
- Full benchmark has **6245 records** across 8 datasets
- Temporary files are automatically cleaned up after evaluation
- LLM judge can be skipped with `--skip-llm-judge` if scores pre-computed
