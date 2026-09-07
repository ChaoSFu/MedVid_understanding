# E-VQA (ST-Evidence-7B) on MedVidU

External frozen-inference baseline adapters for MedVidU STG, RC, and CVS.

Scientific labels:

- STG: E-VQA (ST-Evidence-7B), adapted to MedVidU STG
- RC: E-VQA (ST-Evidence-7B), region-conditioned adaptation
- CVS: E-VQA (ST-Evidence-7B), zero-shot structured reasoning

This package keeps `third_party/EVQA` upstream-clean. It does not download the model, finetune on MedVidU, create fake MP4s, access source videos beyond benchmark-provided frames, or tune prompts/parameters against MedVidU scores.

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

