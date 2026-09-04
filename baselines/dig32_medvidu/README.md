# Fixed DIG-32 + Multiple Qwen Backbones on MedVidU

This baseline enforces a two-stage protocol:

1. Stage A runs the official `Jialuo-Li/DIG` selector once and freezes `dig32_selector_manifest.jsonl`.
2. Stage B runs target Qwen backbones from the frozen manifest only.

The target runners never import DIG selector modules. They read the same selected logical MedVidU positions, timestamps, and frame paths for every backbone, and write `selector_manifest_sha256` plus `selected_positions_hash` into each prediction row.

## Prepare Smoke Inputs

```bash
python -m baselines.dig32_medvidu.run_smoke --run-tests
```

If the official query identifier is unavailable, the smoke report stops with `QUERY_IDENTIFIER_UNAVAILABLE`. Do not substitute task rules or another VLM without review.

## Run Stage A

Start the official DIG query LLM first:

```bash
export MODEL_NAME=Qwen/Qwen3-Next-80B-A3B-Instruct
bash third_party/DIG/scripts/launch_llm.sh
```

Start the fixed reward LMM server:

```bash
export MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
bash third_party/DIG/scripts/launch_mllm.sh
```

Then run selector smoke:

```bash
python -m baselines.dig32_medvidu.run_selector \
  --manifest outputs/baselines/dig32_medvidu/selector/gt_free_manifest.smoke.jsonl \
  --dig-repo-dir third_party/DIG
```

## Run Stage B

Use the same selector manifest and SHA for each target:

```bash
SHA=$(cut -d ' ' -f 1 outputs/baselines/dig32_medvidu/selector/dig32_selector_manifest.sha256)

python -m baselines.dig32_medvidu.run_target --target qwen3vl_4b --selector-sha256 "$SHA"
python -m baselines.dig32_medvidu.run_target --target qwen3vl_8b --selector-sha256 "$SHA"
python -m baselines.dig32_medvidu.run_target --target qwen35_4b --selector-sha256 "$SHA"
python -m baselines.dig32_medvidu.run_target --target qwen38_27b --selector-sha256 "$SHA" --base-url http://127.0.0.1:30000/v1
```

Region Caption is marked `RC_SELECTOR_ADAPTATION_REQUIRED` because MedVidU provides a mandatory anchor frame and bounding box.
