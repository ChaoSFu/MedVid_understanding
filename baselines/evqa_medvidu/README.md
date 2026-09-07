# E-VQA (ST-Evidence-7B) on MedVidU

External frozen-inference baseline adapters for MedVidU STG, RC, and CVS.

Scientific labels:

- STG: E-VQA (ST-Evidence-7B), adapted to MedVidU STG
- RC: E-VQA (ST-Evidence-7B), region-conditioned adaptation
- CVS: E-VQA (ST-Evidence-7B), zero-shot structured reasoning

This package keeps `third_party/EVQA` upstream-clean. It does not download the model, finetune on MedVidU, create fake MP4s, access source videos beyond benchmark-provided frames, or tune prompts/parameters against MedVidU scores.

The default output root is versioned as `evqa_medvidu_v4_stg_target_aligned`.
CVS and RC use fixed task-format contracts: CVS emits the three required scores
on one line, and RC emits a concise textual description of the supplied
reference region. CVS and RC also forbid the `<|seg|>` decoding token. This separates them from STG, whose native output includes
the `<|seg|>` control token and SAM2 mask propagation.

Run order:

```bash
python -B -m baselines.evqa_medvidu.run_preflight
python -B -m baselines.evqa_medvidu.run_official_reproduction \
  --model-path /local/path/ST-Evidence-7B \
  --data-file /path/to/official/st_evidence_final.csv \
  --video-dir /path/to/official/videos \
  --samples 5
python -B -m baselines.evqa_medvidu.run_smoke --model-path /local/path/ST-Evidence-7B
```

After reviewing all v4 smoke artifacts, run the resumable full sequence:

```bash
python -B -m baselines.evqa_medvidu.run_full \
  --model-path /local/path/ST-Evidence-7B \
  --allow-without-official-reproduction \
  --i-reviewed-smoke-and-freeze-adapters
```

The default order is STG (780), RC (310), then CVS (600). Successful
MedVidU predictions are the completion cache, so restarting the same command
only runs missing rows. Per-sample exceptions are written to
`predictions/errors/<task>.jsonl` and do not stop the task. Progress is updated
atomically after every row in `audit/full_run_progress.json`; the final report
and post-inference evaluation artifacts are written under `audit/` and
`evaluation/` respectively.

Visual input policy: the MedVidU adapter passes every benchmark-provided frame
to the model, in listed order, with no `fps` or `max_frames` temporal cap. This
differs from the upstream ST-Evidence script's `fps=1.0`, `max_frames=128`
policy and is recorded as a MedVidU-specific adaptation. For image-list input,
E-VQA's vision utility interprets `max_pixels` as a cap on each frame, so this
adapter fixes it at
`256 * 28 * 28` pixels per frame (approximately 448x448 for square images).
This prevents long, high-resolution clips from exceeding the model context
without dropping frames. The value is recorded in the cache key and prediction
artifacts; override it only as an explicit new, frozen experiment setting.

STG time alignment is a frozen, GT-free adapter. It parses the requested
`start`, `end`, and sampling interval from the human task question, maps source
frame indices to clip-local time using the dataset source timebase, and exports
only the requested timestamp keys with boxes from the nearest benchmark frame.
CholecTrack20 uses its 25 Hz source timebase; CoPESD and EgoSurgery retain their
respective 1 Hz and 0.5 Hz source timebases. The manifest writes
`audit/stg_target_timestamp_alignment.json`, while STG smoke reports requested,
emitted, and missing target boxes. This is an output adapter only: all
benchmark-provided frames are still passed to the model in their listed order.
