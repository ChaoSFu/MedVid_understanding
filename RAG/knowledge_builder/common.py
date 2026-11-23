"""Small portable IO boundary; no ReliVE dataset, answer or certificate imports."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)


def stable_hash(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def strict_loads(text):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('DUPLICATE_JSON_KEY')
            out[k]=v
        return out
    def invalid(_):raise ValueError('NONFINITE_JSON')
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid)


def read_json(path):return strict_loads(Path(path).read_text(encoding='utf-8'))


def write_json(path,value,*,jsonl=False):
    """Atomic create-only writes; identical repeated writes are safe."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    body=(''.join(canonical(v)+'\n' for v in value) if jsonl else canonical(value)+'\n').encode()
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.pending-')
    try:
        with os.fdopen(fd,'wb') as f:f.write(body);f.flush();os.fsync(f.fileno())
        try:os.link(tmp,path)
        except FileExistsError:
            if path.read_bytes()!=body:raise ValueError('IMMUTABLE_OUTPUT_EXISTS:'+str(path)) from None
    finally:os.unlink(tmp)


@contextmanager
def run_lock(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with (root/'.build.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def norm(text):return ' '.join(text.split())
