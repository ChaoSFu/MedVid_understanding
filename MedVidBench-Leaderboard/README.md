---
title: MedVidBench Leaderboard
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: true
license: apache-2.0
short_description: MedVidBench Benchmark Leaderboard - 8 medical video tasks
sdk_version: 5.50.0
tags:
- leaderboard
- medical
- video-understanding
- surgical-ai
---

# MedVidBench Leaderboard

Interactive leaderboard for evaluating Video-Language Models on the **MedVidBench benchmark** - 8 medical video understanding tasks across 8 surgical datasets.

🏆 **Live Demo**: [huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard](https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard)

📄 **Paper**: [arXiv:2512.06581](https://arxiv.org/abs/2512.06581)

## Overview

This leaderboard provides a centralized platform for researchers to:
- **Submit** inference results on the MedVidBench test set
- **Automatically evaluate** across 8 diverse tasks
- **Compare** model performance on standardized metrics
- **Track** state-of-the-art progress in medical video understanding

## Features

### 🎯 8 Medical Video Tasks

| Task | Metric | Description |
|------|--------|-------------|
| **TAL** | mAP@0.5 | Temporal Action Localization - identify start/end times of surgical actions |
| **STG** | mIoU | Spatiotemporal Grounding - locate actions in space (bbox) and time |
| **Next Action** | Accuracy | Predict the next surgical step |
| **DVC** | LLM Judge | Dense Video Captioning - detailed segment descriptions |
| **VS** | LLM Judge | Video Summary - summarize entire surgical videos |
| **RC** | LLM Judge | Region Caption - describe regions indicated by bounding boxes |
| **Skill Assessment** | Accuracy | Evaluate surgical skill levels (JIGSAWS) |
| **CVS Assessment** | Accuracy | Clinical variable scoring |

### ⚙️ Automatic Evaluation

The leaderboard integrates directly with the MedVidBench evaluation pipeline:
- **Validation**: Checks results file format and sample count
- **Execution**: Runs `evaluate_all_pai.py` with dataset-agnostic grouping
- **Parsing**: Extracts task-specific metrics from evaluation output
- **Ranking**: Computes normalized average score across all tasks

### 📊 Test Set Statistics

- **Total samples**: 6,245
- **Source datasets**: 8 (AVOS, CholecT50, CholecTrack20, Cholec80_CVS, CoPESD, EgoSurgery, NurViD, JIGSAWS)
- **Video frames**: ~103,742

## Submission Guide

### 1. Run Inference

Run your model on the MedVidBench test set (6,245 samples) to generate predictions for all 8 tasks.

### 2. Expected Results Format

The leaderboard supports **two formats** for submission:

#### Format 1: Full Format (with Ground Truth)

```json
[
  {
    "question": "<video>\nQuestion text...",
    "response": "Your model's answer",
    "ground_truth": "Correct answer",
    "qa_type": "tal",
    "metadata": {
      "video_id": "...",
      "fps": "1.0",
      ...
    },
    "data_source": "AVOS",
    ...
  },
  ...
]
```

#### Format 2: Prediction-Only Format

```json
[
  {
    "id": "video_id&&start_frame&&end_frame&&fps",
    "qa_type": "tal",
    "prediction": "Your model's answer"
  },
  ...
]
```

**Example**:
```json
[
  {
    "id": "kcOqlifSukA&&22425&&25124&&1.0",
    "qa_type": "tal",
    "prediction": "22.0-78.0, 89.0-94.0 seconds."
  },
  {
    "id": "VsKw5d-4rq8&&13561&&16184&&1.0",
    "qa_type": "stg",
    "prediction": "[10, 20, 30, 40] 5.0-10.0 seconds."
  }
]
```

**Key differences**:
- Format 1: Uses `response` + `ground_truth` fields with full metadata (dictionary format indexed by string keys "0", "1", etc.)
- Format 2: Uses `id` + `prediction` fields only (list format, GT merged automatically by **index position**)
- The `id` field format: `{video_id}&&{start_frame}&&{end_frame}&&{fps}` is included for reference but **matching is done by array index**
- **Important**: Predictions in Format 2 must be in the same order as the test set

**Valid qa_types**:
- `tal` - Temporal Action Localization
- `stg` - Spatiotemporal Grounding
- `next_action` - Next Action Prediction
- `dense_captioning` - Dense Video Captioning
- `video_summary` - Video Summary
- `region_caption` - Region Caption
- `skill_assessment` - Skill Assessment (JIGSAWS)
- `cvs_assessment` - CVS Assessment

### 3. Upload to Leaderboard

1. Visit the [leaderboard](https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard)
2. Go to the **Submit Results** tab
3. Fill in:
   - **Model Name** (e.g., "Qwen2.5-VL-7B-MedVidBench")
   - **Organization** (e.g., "Your University")
   - **Contact** (optional)
4. Upload your results JSON file
5. Click **Submit to Leaderboard**

The system will:
- Validate your file (format + sample count)
- Run automatic evaluation (~2-5 minutes with `--skip-llm-judge`, ~10-20 minutes with LLM judge)
- Extract metrics for all 8 tasks
- Add your model to the leaderboard

**Note**: By default, DVC/VS/RC are evaluated with `--skip-llm-judge` for faster results (caption metrics will be 0.0). You can run LLM judge evaluation later using the button on the leaderboard page.

### 4. Run LLM Judge Evaluation (Optional)

If your submission was evaluated with `--skip-llm-judge` (DVC_llm, VS_llm, RC_llm are all 0.0), you can compute these metrics later:

1. Go to the **Leaderboard** tab
2. Scroll to the **"Run LLM Judge Evaluation"** section
3. Enter your model name (exact match)
4. Click **"Start Evaluation"**

The system will:
- Start evaluation in the background (runs independently)
- Re-run evaluation for DVC/VS/RC tasks with LLM judge (GPT-4.1/Gemini)
- Automatically update your leaderboard entry when complete
- Preserve all other metrics (TAL, STG, NAP, SA, CVS)

**✅ Background Execution**:
- You can **close the browser** after starting - evaluation continues running
- Come back later and click **"Check Status"** to see progress
- The leaderboard will be automatically updated when complete

**Time**: ~10-20 minutes depending on API rate limits

**Availability**: Only available when ALL three caption metrics are 0.0

**How to Check Status**:
1. Enter the same model name
2. Click **"Check Status"** button
3. View recent logs and progress
4. Or simply refresh the leaderboard to see if metrics are updated

## Evaluation Metrics

### Task-Specific Metrics

| Task | Metric Extracted | Details |
|------|------------------|---------|
| TAL | `mAP@0.5` | Mean Average Precision at IoU=0.5 |
| STG | `mean_iou` | Mean Intersection over Union (spatial + temporal) |
| Next Action | `Weighted Average Accuracy` | Classification accuracy |
| DVC/VS/RC | `Average LLM Judge Score` | Average of R2, R4, R5, R7, R8 (1-5 scale) |
| Skill/CVS | `accuracy` | Classification accuracy |

### LLM Judge Details

For caption tasks (DVC, VS, RC), we use **GPT-4.1** or **Gemini-Pro** with rubric-based scoring:

**5 Key Aspects** (1-5 scale each):
- **R2**: Relevance & Medical Terminology
- **R4**: Actionable Surgical Actions
- **R5**: Comprehensive Detail Level
- **R7**: Anatomical & Instrument Precision
- **R8**: Clinical Context & Coherence

**Final score** = Average of R2, R4, R5, R7, R8

### Score Normalization

To compute the **average score** fairly across tasks:
1. **LLM Judge scores** (1-5 scale) are normalized: `(score - 1) / 4` → [0, 1]
2. **Other metrics** (already 0-1) remain unchanged
3. **Average** = mean of all 8 normalized task scores

## Links

- 📄 **Paper**: [https://arxiv.org/abs/2512.06581](https://arxiv.org/abs/2512.06581)
- 🌐 **Project**: [https://uii-ai.github.io/MedGRPO/](https://uii-ai.github.io/MedGRPO/)
- 💾 **Dataset**: [https://huggingface.co/datasets/UII-AI/MedVidBench](https://huggingface.co/datasets/UII-AI/MedVidBench)
- 💻 **GitHub**: [https://github.com/UII-AI/MedGRPO-Code](https://github.com/UII-AI/MedGRPO-Code)
- 🎮 **Demo**: [https://huggingface.co/spaces/UII-AI/MedGRPO-Demo](https://huggingface.co/spaces/UII-AI/MedGRPO-Demo)
- 🏆 **Leaderboard**: [https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard](https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard)

## Citation

```bibtex
@article{su2024medgrpo,
  title={MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding},
  author={Su, Yuhao and Choudhuri, Anwesa and Gao, Zhongpai and Planche, Benjamin and Nguyen, Van Nguyen and Zheng, Meng and Shen, Yuhan and Innanje, Arun and Chen, Terrence and Elhamifar, Ehsan and Wu, Ziyan},
  journal={arXiv preprint arXiv:2512.06581},
  year={2025}
}
```

## License

- **Leaderboard Code**: Apache 2.0
- **Dataset**: CC BY-NC-SA 4.0 (Non-commercial, Share-alike)

## Contact

For questions or issues:
- Open an issue on [GitHub](https://github.com/UII-AI/MedGRPO-Code)
- Visit the [project page](https://uii-ai.github.io/MedGRPO/)