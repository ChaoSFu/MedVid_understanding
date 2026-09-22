from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from relive.v2.protocol import PROTOCOL_VERSION, ProtocolManifest
from relive.v2.protocol_manifests import write_manifest


class ProtocolCliTests(unittest.TestCase):
    def test_split_validator_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); paths = []
            for split in ("protocol_dev", "calibration", "blind_test"):
                path = root / f"{split}.jsonl"
                token = lambda suffix: hashlib.sha256((split + suffix).encode()).hexdigest()
                write_manifest(path, [ProtocolManifest(PROTOCOL_VERSION, token("program"), split, token("video"), (token("frame"),), "automatic")])
                paths.append(path)
            script = Path(__file__).resolve().parents[1] / "scripts" / "validate_relive_v2_protocol_manifests.py"
            result = subprocess.run([sys.executable, str(script), "--protocol-dev", str(paths[0]), "--calibration", str(paths[1]), "--blind-test", str(paths[2])],
                                    check=True, text=True, capture_output=True, env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")})
            self.assertEqual(json.loads(result.stdout)["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
