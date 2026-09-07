from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Guarded full-run entrypoint for E-VQA MedVidU.")
    parser.add_argument("--i-reviewed-smoke-and-freeze-adapters", action="store_true")
    args = parser.parse_args()
    if not args.i_reviewed_smoke_and_freeze_adapters:
        print("STOP: full runs are intentionally disabled until official reproduction, STG/RC/CVS smoke, cache rerun, and audits are manually reviewed.")
        return 2
    print("Full-run orchestration is intentionally staged; run STG first, then RC, then CVS after adapter freeze.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

