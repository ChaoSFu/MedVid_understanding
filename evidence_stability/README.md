# MedVidU Evidence Stability

Training-free Evidence Stability experiments for MedVidU medical video temporal
action localization.

This Phase B implementation only covers:

- TAL sample normalization
- dataset-aware frame-to-local-time adapters from Phase B.1
- temporal frame cells for GT visibility accounting
- position-based sliding window generation
- diagnostic GT alignment metrics for audit outputs

It intentionally does not run VLM inference.

Phase B.1 controls:

- GT temporal spans are not used for candidate proposal.
- `windows.jsonl` does not contain GT alignment fields.
- `windows.jsonl` is limited to analysis-eligible windows for future VLM probes.
- GT alignment metrics are written only to diagnostic audit files.
- GT is clip-local and is never shifted by `input_video_start_time`.
- All downstream primary IDs use `qa_id`; the original MedVidU id is stored as
  `clip_id`.
