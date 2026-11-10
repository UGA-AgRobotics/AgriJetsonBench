#!/usr/bin/env python3
import argparse, csv, hashlib, pathlib, sys

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest', default='MANIFEST.csv')
    args=ap.parse_args()
    root=pathlib.Path(args.manifest).resolve().parent
    bad=[]; total=0
    with open(args.manifest, newline='', encoding='utf-8') as f:
        r=csv.DictReader(f)
        for row in r:
            rel=row.get('relative_path','')
            expected=row.get('sha256','')
            if not rel or not expected: continue
            p=root/rel
            if not p.exists():
                bad.append((rel,'missing'))
                continue
            got=sha256(p)
            total += 1
            if got.lower()!=expected.lower(): bad.append((rel,'sha mismatch'))
    if bad:
        print('FAILED checksum validation')
        for rel,msg in bad[:100]: print(rel,msg)
        sys.exit(1)
    print(f'OK: verified {total} files')
if __name__ == '__main__': main()
