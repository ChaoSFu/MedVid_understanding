# ReliVE v2 protocol_dev 9-case contracts

Each file in `protocol_dev/cases/` is a portable
`relive-v2-protocol-dev-case-v1` case. It contains no absolute data-root path
and no oracle coordinate. Oracle coordinates, when humans create them, must be
stored outside this directory; a case can reference only an artifact name and
its readiness state. `assert_automatic_certificate_input_safe()` rejects
oracle fields before any automatic certificate path.

The source draft is treated as a human draft, not a runtime input. The current
draft had concatenated top-level objects and a full-width comma. The normalizer
records those facts in every affected case provenance; it does not rewrite the
draft or turn missing review fields into an admission.

```bash
cd /home/huihui/codes/MedVid_understanding/relive
conda activate MedVidU-cu124

# Schema-only: no data root, no model, no cache, no certificate.
PYTHONPATH=src python scripts/audit_v2_protocol_dev_cases.py \
  --cases-dir protocol_dev/cases \
  --output-dir /mnt/hdd/huihui/MedVid_understanding/relive_output/real/protocol_dev_cases_audit_$(date +%Y%m%d_%H%M%S)
```

For an optional existence check, create a local JSON mapping that is not
committed, e.g. `configs/v2/protocol_dev_data_roots.local.json`, from the
example. It maps `*_ROOT` keys to a local dataset root and never changes a case
file.

```bash
PYTHONPATH=src python scripts/audit_v2_protocol_dev_cases.py \
  --cases-dir protocol_dev/cases \
  --oracle-dir /path/to/separately-stored-human-oracle-artifacts \
  --data-roots configs/v2/protocol_dev_data_roots.local.json \
  --output-dir /path/to/new_audit
```

`--strict` exits nonzero whenever any case is `PENDING` or `INVALID`. It is
appropriate only after humans supply canonical source-record hashes, reviewer
identities/decisions, portable frame patterns, missing predicate components,
and externally stored oracle references.
