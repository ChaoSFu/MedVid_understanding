# MedVidU Evidence Stability

Training-free Evidence Stability experiments for MedVidU medical video temporal
action localization.

This Phase B implementation only covers:

- TAL sample normalization
- endpoint-anchored Mapping C frame-to-local-time conversion
- temporal frame cells for GT visibility accounting
- position-based sliding window generation
- diagnostic GT alignment metrics for audit outputs

It intentionally does not run VLM inference.

Scientific controls:

- GT temporal spans are not used for candidate proposal.
- `windows.jsonl` does not contain GT alignment fields.
- GT alignment metrics are written only to diagnostic audit files.
- GT is clip-local and is never shifted by `input_video_start_time`.
