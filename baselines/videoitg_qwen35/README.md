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

For full runs, shard selector across GPUs instead of sending VideoITG through a
chat-serving engine. Each process writes its own JSONL shard, then merge them:

```bash
SEL_ROOT=outputs/baselines/videoitg_medvidu_top32
VIDEOITG_REPO=/path/to/VideoITG

CUDA_VISIBLE_DEVICES=0 nohup python -m baselines.videoitg_qwen35.run_selector \
  --manifest $SEL_ROOT/manifest/medvidu_videoitg_manifest_gt_free.jsonl \
  --output-root $SEL_ROOT \
  --videoitg-repo-dir $VIDEOITG_REPO \
  --device cuda:0 \
  --num-shards 4 \
  --shard-index 0 \
  > $SEL_ROOT/logs/videoitg_selector_shard0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -m baselines.videoitg_qwen35.run_selector \
  --manifest $SEL_ROOT/manifest/medvidu_videoitg_manifest_gt_free.jsonl \
  --output-root $SEL_ROOT \
  --videoitg-repo-dir $VIDEOITG_REPO \
  --device cuda:0 \
  --num-shards 4 \
  --shard-index 1 \
  > $SEL_ROOT/logs/videoitg_selector_shard1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -m baselines.videoitg_qwen35.run_selector \
  --manifest $SEL_ROOT/manifest/medvidu_videoitg_manifest_gt_free.jsonl \
  --output-root $SEL_ROOT \
  --videoitg-repo-dir $VIDEOITG_REPO \
  --device cuda:0 \
  --num-shards 4 \
  --shard-index 2 \
  > $SEL_ROOT/logs/videoitg_selector_shard2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -m baselines.videoitg_qwen35.run_selector \
  --manifest $SEL_ROOT/manifest/medvidu_videoitg_manifest_gt_free.jsonl \
  --output-root $SEL_ROOT \
  --videoitg-repo-dir $VIDEOITG_REPO \
  --device cuda:0 \
  --num-shards 4 \
  --shard-index 3 \
  > $SEL_ROOT/logs/videoitg_selector_shard3.log 2>&1 &
```

After every shard finishes:

```bash
python -m baselines.videoitg_qwen35.run_merge_selector_shards \
  --output-root $SEL_ROOT \
  --num-shards 4
```

The merged files are the standard downstream inputs:

```text
$SEL_ROOT/selector/videoitg_top32.jsonl
$SEL_ROOT/selector/videoitg_scores.jsonl
$SEL_ROOT/selector/selector_errors.jsonl
```

If a `nohup` selector run is interrupted while writing, repair the partial JSONL
before resuming:

```bash
python -m baselines.videoitg_qwen35.run_repair_jsonl \
  $SEL_ROOT/selector/videoitg_top32.jsonl \
  $SEL_ROOT/selector/videoitg_scores.jsonl \
  $SEL_ROOT/selector/selector_errors.jsonl \
  --report $SEL_ROOT/selector/jsonl_repair_report.json
```

The repair command writes `*.corrupt.bak` backups and keeps only complete JSON
object lines.

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
