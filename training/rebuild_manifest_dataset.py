"""Portable materialization from a verified image/label manifest, no resampling."""
import argparse
import json
import os
import shutil
from pathlib import Path,PureWindowsPath
from prepare_data import parse_label
import hashlib
ROOT=Path(__file__).resolve().parent
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,default=ROOT/'manifest.json')
    p.add_argument('--raw-root',type=Path,required=True); p.add_argument('--output',type=Path,default=ROOT/'data/rebuilt1200')
    args=p.parse_args(); output=args.output.resolve()
    if output.exists(): raise ValueError('Use a new project-local data directory')
    records=json.loads(args.manifest.read_text()); index={}
    for flight in sorted({m['flight'] for m in records}):
        directory=args.raw_root/flight
        index[flight]=({p.name:p for p in directory.rglob('*.jpg')},{p.name:p for p in directory.rglob('*.txt')})
    for m in records:
        name=PureWindowsPath(m['image']).name; label_name=PureWindowsPath(m['label']).name
        image=index[m['flight']][0][name]; label=index[m['flight']][1][label_name]
        if digest(image)!=m['image_sha256'] or digest(label)!=m['label_sha256']: raise ValueError('Source hash mismatch')
        destination=output/'images'/m['split']/PureWindowsPath(m['derived_image']).name
        target=output/'labels'/m['split']/destination.with_suffix('.txt').name
        destination.parent.mkdir(parents=True,exist_ok=True); target.parent.mkdir(parents=True,exist_ok=True)
        try: os.link(image,destination)
        except OSError: shutil.copy2(image,destination)
        rows=parse_label(label); text='\n'.join(' '.join(f'{x:.9g}' for x in row) for row in rows)+('\n' if rows else '')
        target.write_text(text,encoding='utf-8')
        if digest(target)!=m['derived_label_sha256']: raise ValueError('Derived label differs')
    yaml=f'path: {output.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: gate\nkpt_shape: [4, 3]\n'
    (output/'dataset.yaml').write_text(yaml,encoding='utf-8')
    (output/'manifest.json').write_text(args.manifest.read_text(),encoding='utf-8')
    print(json.dumps(dict(materialized=len(records),yaml=str(output/'dataset.yaml'),all_source_and_label_hashes_verified=True)))


if __name__=='__main__': main()
