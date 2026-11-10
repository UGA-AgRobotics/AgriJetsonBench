#!/usr/bin/env python3
"""Check added records; no model execution or new measurements."""
from pathlib import Path
import csv, json, hashlib, subprocess, sys
from collections import Counter
R=Path(__file__).resolve().parent

def main():
    subprocess.run([sys.executable,str(R/'verify_evidence.py')],check=True)
    subprocess.run([sys.executable,str(R/'diagnostics/recovery_addendum/verify_diagnostics.py')],check=True)
    expected={'fp16':'ccb9b393ce5f50a8c2eb465d84988fe008b03392e826db4773d4e4a0e8ab91ae','int8':'67b2e2e9b5eff8a6c91a0d67a081858c846ac35f19744813402b938363514929'}
    for precision,h in expected.items():
        p=R/f'accuracy/recovered_AGX_predictions/yolo11s_{precision}_detections_coco.json'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==h
        records=json.loads(p.read_text());assert records and all('image_id' in d for d in records)
    for board,count,key in [('AGX',193,'xmma_tactic_family'),('Nano',199,'tactic_family')]:
        d=R/f'fallback/diagnostic_{board}'
        layers=json.loads((d/'detailed_inspector.json').read_text())['Layers']
        rows=list(csv.DictReader((d/'layer_precision.csv').open()))
        assert len(layers)==len(rows)==count
        families=Counter(r[key] or 'unknown' for r in rows)
        expected=({'FP16':52,'INT8':5,'TF32':1,'unknown':135} if board=='AGX' else {'FP16':49,'INT8':8,'TF32':7,'FP32':2,'unknown':133})
        assert dict(families)==expected,(board,families)
    print('PASS: recovered predictions, diagnostics inputs, and both diagnostic-plan export summaries.')
    print('Tactic families use the source reports naming rules; no historical-plan precision is inferred.')
if __name__=='__main__':main()
