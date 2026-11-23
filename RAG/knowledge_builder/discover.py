"""Discover every PDF, including nested/Unicode paths and explicit failures."""
from pathlib import Path
from .common import file_hash,stable_hash


def discover(root):
    root=Path(root).expanduser().resolve()
    if not root.is_dir():raise ValueError('REFERENCES_DIRECTORY_MISSING')
    paths=[];errors=[]
    def on_error(exc):errors.append({'path':str(exc.filename),'error':'DIRECTORY_UNREADABLE'})
    import os
    for folder,dirs,files in os.walk(root,followlinks=False,onerror=on_error):
        for d in list(dirs):
            if (Path(folder)/d).is_symlink():
                errors.append({'path':str((Path(folder)/d).relative_to(root)),'error':'SYMLINK_DIRECTORY_NOT_FOLLOWED'})
                dirs.remove(d)
        paths.extend(Path(folder)/n for n in files if Path(n).suffix.lower()=='.pdf')
    sources=[]
    for path in sorted(paths,key=lambda p:p.relative_to(root).as_posix()):
        rel=path.relative_to(root).as_posix()
        try:
            if not path.resolve().is_relative_to(root):raise ValueError('PDF_SYMLINK_ESCAPES_REFERENCES')
            sha=file_hash(path);sid='src_'+stable_hash({'sha256':sha,'path':rel})[:24]
            sources.append({'source_id':sid,'file_path':rel,'file_name':path.name,'sha256':sha,
                            'file_hash':sha,'processing_status':'DISCOVERED','file_bytes':path.stat().st_size})
        except (OSError,ValueError) as exc:
            sources.append({'source_id':'unreadable_'+stable_hash(rel)[:24],'file_path':rel,'file_name':path.name,
                            'sha256':None,'processing_status':'FAILED','failure_reason':str(exc) if isinstance(exc,ValueError) else type(exc).__name__})
    return root,sources,errors
