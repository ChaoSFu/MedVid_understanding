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

The official DIG launch scripts use vLLM. On CUDA 12.4, this baseline also
supports an OpenAI-compatible SGLang serving adaptation. It keeps the official
DIG prompts, model IDs, CAFS, reward semantics, and refinement untouched, but
records `sglang` in the selector manifest and provenance.

Set the local checkpoint paths. The query model must be the exact fixed
checkpoint; do not replace it with a smaller or quantized model.

```bash
QUERY_MODEL=/path/to/Qwen3-Next-80B-A3B-Instruct
REWARD_MODEL=/path/to/Qwen3-VL-8B-Instruct
```

Start the SGLang query server. Set `--tp-size` to the number of GPUs assigned
to this server, and reserve different GPUs for the reward service.

```bash
python -m sglang.launch_server \
  --model-path "$QUERY_MODEL" \
  --served-model-name Qwen/Qwen3-Next-80B-A3B-Instruct \
  --host 127.0.0.1 --port 30000 --api-key token-abc123 \
  --tp-size 8 --mem-fraction-static 0.85
```

Start the fixed reward LMM in a second terminal:

```bash
python -m sglang.launch_server \
  --model-path "$REWARD_MODEL" \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 --port 30001 --api-key token-abc123 \
  --tp-size 1 --mem-fraction-static 0.80
```

Run the selector with the SGLang endpoint and record the serving adaptation:

```bash
python -m baselines.dig32_medvidu.run_selector \
  --manifest outputs/baselines/dig32_medvidu/selector/gt_free_manifest.smoke.jsonl \
  --dig-repo-dir third_party/DIG \
  --query-base-url http://127.0.0.1:30000/v1 \
  --reward-base-url http://127.0.0.1:30001/v1 \
  --query-serving-backend sglang \
  --reward-serving-backend sglang
```

### Official vLLM Alternative

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
