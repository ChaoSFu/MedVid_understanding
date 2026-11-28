"""Immutable manifests shared by freeze, retrieval and paired planning."""
from pathlib import Path
from RAG.knowledge_builder.common import canonical,stable_hash,file_hash,read_json,write_json,run_lock

VERSION='relive-stage1-rag-v1'


def new_output(path):
    root=Path(path)
    if root.exists():raise ValueError('NEW_OUTPUT_DIRECTORY_REQUIRED')
    root.mkdir(parents=True)
    return root


def separate_output(path,*input_dirs):
    """A derived directory cannot be placed inside a sealed input (or vice versa)."""
    output=Path(path).resolve()
    for item in input_dirs:
        source=Path(item).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('OUTPUT_OVERLAPS_IMMUTABLE_INPUT')


def seal(root,kind,**metadata):
    root=Path(root)
    manifest={'format':kind,**metadata,'artifacts':{
        p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob('*'))
        if p.is_file() and not p.name.startswith('.') and p.name!='manifest.json'}}
    write_json(root/'manifest.json',manifest)
    return manifest


def verify(root,kind=None):
    root=Path(root).resolve();m=read_json(root/'manifest.json')
    if kind and m['format']!=kind:raise ValueError('MANIFEST_KIND_MISMATCH')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*')
            if p.is_file() and not p.name.startswith('.') and p.name!='manifest.json'}
    if actual!=set(m['artifacts']):raise ValueError('UNMANIFESTED_ARTIFACT')
    for name,sha in m['artifacts'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or file_hash(p)!=sha:raise ValueError('ARTIFACT_HASH_MISMATCH')
    return m


def jsonl(path):
    from RAG.knowledge_builder.common import strict_loads
    return [strict_loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
