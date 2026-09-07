# E-VQA (ST-Evidence-7B) on MedVidU

External frozen-inference baseline adapters for MedVidU STG, RC, and CVS.

Scientific labels:

- STG: E-VQA (ST-Evidence-7B), adapted to MedVidU STG
- RC: E-VQA (ST-Evidence-7B), region-conditioned adaptation
- CVS: E-VQA (ST-Evidence-7B), zero-shot structured reasoning

This package keeps `third_party/EVQA` upstream-clean. It does not download the model, finetune on MedVidU, create fake MP4s, access source videos beyond benchmark-provided frames, or tune prompts/parameters against MedVidU scores.

The default output root is versioned as `evqa_medvidu_v2_text_contracts`.
CVS and RC use fixed task-format contracts: CVS emits the three required scores
on one line, and RC emits a concise textual description of the supplied
reference region. This separates them from STG, whose native output includes
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

Full runs are intentionally guarded until smoke artifacts and cache restart audits are reviewed.

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
