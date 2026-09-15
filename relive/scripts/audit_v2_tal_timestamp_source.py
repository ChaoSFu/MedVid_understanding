#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from relive.v2.timestamp_provenance import TimestampProvenanceError, export_public_timestamps

def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a registered public timestamp source without inference.")
    parser.add_argument("--public-media-projection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--documented-timebase-source", type=Path)
    parser.add_argument("--public-video", type=Path)
    parser.add_argument("--decoder-frame-map", type=Path)
    args = parser.parse_args()
    try:
        result = export_public_timestamps(media_projection_path=args.public_media_projection, output_dir=args.output_dir,
            documented_source_path=args.documented_timebase_source, public_video_path=args.public_video,
            decoder_frame_map_path=args.decoder_frame_map)
    except TimestampProvenanceError as exc:
        print(f"ReliVE-v2 TAL timestamp-source audit error: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
