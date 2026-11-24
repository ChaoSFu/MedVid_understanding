"""Write-once run/checkpoint helpers shared with the existing artifact format."""
from pathlib import Path
from relive.claim_revalidation.offline import read_json, write, digest
from relive.storage.artifacts import stable_hash


def verify(root):
    root = Path(root); m = read_json(root / 'manifest.json')
    if stable_hash({k: v for k, v in m.items() if k != 'manifest_sha256'}) != m['manifest_sha256']:
        raise ValueError('MANIFEST_INVALID')
    for name, h in m['files'].items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or digest(p) != h: raise ValueError('ARTIFACT_HASH_INVALID')
    return m


def checkpoint(path, identity):
    cp = read_json(path)
    if cp['identity'] != identity or cp['sha256'] != stable_hash(cp['result']): raise ValueError('CHECKPOINT_INVALID')
    return cp['result']


def save(path, result, identity):
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, {'identity': identity, 'result': result, 'sha256': stable_hash(result)})


def seal(root, **metadata):
    import subprocess
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent,
                                  capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError): revision = None
    files = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name != '.run.lock'}
    m = {**metadata, 'code_commit': revision, 'files': files}; m['manifest_sha256'] = stable_hash(m)
    write(root / 'manifest.json', m); return m
