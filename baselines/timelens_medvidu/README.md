# TimeLens-8B on MedVidU TAL

This is an external baseline adapter for frozen `TencentARC/TimeLens-8B` on MedVidU temporal action localization.

The experiment is:

`MedVidU TAL sample -> benchmark-provided frame sequence + human question -> official frozen TimeLens-8B grounding inference -> parsed temporal spans -> MedVidBench TAL evaluation`.

It must be described as:

> TimeLens-8B adapted to MedVidU TAL using the benchmark-provided visual frame sequence and the official frozen TimeLens grounding model.

It is not an official TimeLens benchmark result on MedVidU.

## Boundaries

- Uses only `sample["video"]` frame lists from MedVidU.
- Does not access AVOS, CholecT50, CoPESD, EgoSurgery, or NurViD source videos.
- Does not convert frame lists to fake MP4.
- Does not resample MedVidU frames again at TimeLens native raw-video `FPS=2`.
- Preserves original frame order and duplicate logical frames.
- Keeps TimeLens-8B frozen: no fine-tuning, LoRA, calibration, few-shot prompts, or medical prompt engineering.
- Raw predictions do not contain GT spans, GT answers, evidence labels, IoU, accuracy, or intervention metadata.

## Actual Data Root

The Train/Val JSON checked in this repo uses `/root/data/...` paths. Other project artifacts show the intended local data root as:

```bash
/mnt/hdd3/huihui/hh_datas/MedVidU/valdata
```

Set the actual path explicitly when running if your machine mounts it elsewhere:

```bash
export MEDVIDU_DATA_ROOT=/path/to/MedVidU/valdata
```

or pass:

```bash
--new-data-root /path/to/MedVidU/valdata
```

Frame existence is audited before smoke inference. Missing frames are a STOP condition.

## Build GT-Free Manifest

```bash
python -m baselines.timelens_medvidu.manifest \
  --data-path data_json/init_datas/medvidu_eccv2026_trainval.json \
  --old-data-root /root/data \
  --new-data-root "$MEDVIDU_DATA_ROOT" \
  --smoke
```

Outputs include:

- `outputs/baselines/timelens8b_medvidu_tal/manifest/medvidu_tal_timelens_manifest_gt_free.jsonl`
- `outputs/baselines/timelens8b_medvidu_tal/manifest/smoke_sample_ids.json`

## Audits

```bash
python -m baselines.timelens_medvidu.audit_artifacts
python -m baselines.timelens_medvidu.audit_timestamp_metadata
python -m baselines.timelens_medvidu.compare_timestamp_adapters
```

`audit_timestamp_metadata` checks whether Qwen3-VL frame-list metadata can faithfully represent MedVidU clip-local timestamps using the GT-free `effective_fps`. If timestamps are non-uniform or duplicate logical timestamps make a single fps representation inaccurate, the run stops.

`compare_timestamp_adapters` compares two GT-free timestamp adapters on the 20-sample smoke manifest:

- `strict_single_fps_subset`: official TimeLens-8B/Qwen3 frame-list video input with `fps=effective_fps`, restricted to samples whose local timestamps are exactly representable by one fps. This is the most faithful to official TimeLens-8B behavior, but only a diagnostic subset when MedVidU timestamps are non-uniform or duplicated.
- `textual_timestamp_image_sequence`: preserves every MedVidU logical frame as an image and writes the GT-free `local_time` before each frame, followed by the unchanged official TimeLens grounding prompt. This is more faithful to MedVidU timing and coverage, but should be named as an adapted TimeLens variant, not an official TimeLens-8B run.

To also call `qwen_vl_utils.process_vision_info` inside the active environment:

```bash
python -m baselines.timelens_medvidu.compare_timestamp_adapters --process-qwen
```

## Smoke Inference

Use a local checkpoint. The runner refuses to download a missing model path.

```bash
python -m baselines.timelens_medvidu.run_smoke \
  --model-path /local/path/to/TimeLens-8B \
  --smoke
```

Run the same command a second time for the exact cache restart test. The expected restart audit is:

- `completed_cache_at_start = 20`
- `new_inference_count = 0`
- `skipped_cached_count = 20`

## Evaluation

```bash
python -m baselines.timelens_medvidu.evaluate_medvidu_tal \
  --predictions outputs/baselines/timelens8b_medvidu_tal/predictions/timelens8b_tal_smoke_predictions.jsonl \
  --data-path data_json/init_datas/medvidu_eccv2026_trainval.json \
  --split-name smoke
```

Full run command, after smoke and timestamp compatibility pass:

```bash
python -m baselines.timelens_medvidu.run_full \
  --model-path /local/path/to/TimeLens-8B
```

Do not run full TAL until the 20-sample smoke, timestamp compatibility preflight, and exact cache restart test pass.
