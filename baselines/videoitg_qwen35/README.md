# VideoITG-32 + Qwen3.5-4B on MedVidU

This directory implements an external, training-free baseline:

MedVidU sample -> GT-free task input -> frozen VideoITG-8B selector -> Top-32 frames -> chronological reorder -> Qwen3.5-4B prediction -> offline MedVidBench evaluation.

VideoITG answers only "which frames are relevant"; Qwen3.5 answers the downstream MedVidU task. The two stages communicate only through JSONL files and are intended to run in separate Python environments.

## Official VideoITG Contract

The implementation follows the reviewed official VideoITG commit `50a60a822c0e362bfd8747c45ba34e66e9c9d650`:

- checkpoint: `nvidia/VideoITG-8B`
- candidate limit: 512 logical MedVidU frame positions
- relevance score: official `topk_model(...).logits.sigmoid()`
- ranking: score descending
- selection: Top-32, or all candidates when fewer than 32
- downstream order: selected positions sorted chronological ascending

Region Caption is marked selector N/A because MedVidU already supplies local region evidence through `RC_info`.

## First Smoke Run

Prepare GT-free manifest and fixed smoke IDs:

```bash
python -m baselines.videoitg_qwen35.run_smoke \
  --old-data-root /root/data \
  --new-data-root /mnt/hdd3/huihui/hh_datas/MedVidU/valdata
```

Run Qwen3.5 compatibility preflight in the Qwen environment:

```bash
python -m baselines.videoitg_qwen35.run_inference --preflight-only
```

Run VideoITG selector in the separate VideoITG environment:

```bash
python -m baselines.videoitg_qwen35.run_selector \
  --manifest outputs/baselines/videoitg_qwen35/manifest/medvidu_videoitg_manifest_gt_free.smoke.jsonl \
  --videoitg-repo-dir /path/to/VideoITG
```

If your data root differs, regenerate the manifest with the correct
`--new-data-root` before running selector or Qwen inference. The remap changes
only filesystem prefixes; it does not add frames or alter the MedVidU evidence
budget.

Run deterministic Qwen3.5 inference:

```bash
python -m baselines.videoitg_qwen35.run_inference \
  --manifest outputs/baselines/videoitg_qwen35/manifest/medvidu_videoitg_manifest_gt_free.smoke.jsonl \
  --selector outputs/baselines/videoitg_qwen35/selector/videoitg_top32.jsonl
```

For larger served Qwen models, such as Qwen3.8-27B, prefer the SGLang
OpenAI-compatible async runner. Start SGLang separately, then run:

```bash
python -m baselines.videoitg_qwen35.run_inference_sglang \
  --manifest outputs/baselines/videoitg_qwen35/manifest/medvidu_videoitg_manifest_gt_free.smoke.jsonl \
  --selector outputs/baselines/videoitg_qwen35/selector/videoitg_top32.jsonl \
  --model /path/to/local/Qwen3.8-27B \
  --base-url http://127.0.0.1:30000/v1 \
  --output-root outputs/baselines/videoitg_qwen38_27b_sglang \
  --concurrency 4
```

The SGLang runner sends the exact VideoITG-selected frame list as timestamped
image inputs. It does not rerun selection, resample frames, or use ground truth.

Evaluate offline:

```bash
python -m baselines.videoitg_qwen35.run_evaluation \
  --predictions outputs/baselines/videoitg_qwen35/predictions/qwen35_videoitg_predictions.jsonl \
  --ground-truth data_json/init_datas/medvidu_eccv2026_trainval.json
```

Do not run the full MedVidU split until the smoke selector, Qwen inference, parser, and leakage audits pass.
