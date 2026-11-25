# STG data for V4.1 claim generation

This upload contains STG only: **780 records** from the unchanged
`data_json/init_datas/medvidu_eccv2026_trainval.json` (6270-record trainval source).

| Source dataset | Questions |
|---|---:|
| CholecTrack20 | 195 |
| CoPESD | 137 |
| EgoSurgery | 448 |
| Total | 780 |

- `medvidu_trainval_stg.json`: unchanged original records, including evaluation fields.
- `stg_source_record_index.jsonl`: mapping to original full-dataset positions 1700–2479.
- `stg_v41_inputs/`: GT-free runtime projection, provenance, source hashes and audit.

Original full-source SHA-256:
`cee45807f26d8ceeb5d4b3969cd0327dbe4e97e41bff7ed4ed8b5e6be0ba706f`.

The generator reads only public question/task and frozen Stage-1 requirements.
Never send the raw evaluation record/answer/annotations as the model prompt.
Original sample keys are preserved so the existing full Stage-1 bank can be reused.
The TAL and Dense Captioning exports remain local and are not part of this upload.

See `relive/docs/stg_claim_bank_v41.md` for executable server commands and resume/replay.
