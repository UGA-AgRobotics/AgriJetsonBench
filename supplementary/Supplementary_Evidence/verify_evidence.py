#!/usr/bin/env python3
"""Check the distributed records and derived tables without rerunning inference."""
from pathlib import Path
import csv, hashlib, json, math

ROOT=Path(__file__).resolve().parent

def rows(rel):
    with (ROOT/rel).open(newline='') as f:return list(csv.DictReader(f))

def equal(a,b,rel=0,abs_tol=1e-12):
    assert math.isclose(float(a),float(b),rel_tol=rel,abs_tol=abs_tol),(a,b)

def segmentation(report):
    cm=report['metrics']['confusion_matrix']
    iou=[];dice=[]
    for i in range(3):
        tp=cm[i][i];gt=sum(cm[i]);pred=sum(r[i] for r in cm)
        iou.append(tp/(gt+pred-tp));dice.append(2*tp/(gt+pred))
    assert sum(sum(r) for r in cm)==3187752960
    return {'mIoU':sum(iou)/3,'mean Dice':sum(dice)/3,'Crop IoU':iou[1],'Weed IoU':iou[2]}

def main():
    hashes=0
    for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
        digest,rel=line.split('  ',1)
        p=ROOT/rel
        assert p.is_file(),f'Missing file: {rel}'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,f'Hash mismatch: {rel}'
        hashes+=1
    comparisons=rows('accuracy/accuracy_comparison_both_boards.csv')
    assert len(comparisons)==14
    reports={}
    for row in comparisons:
        fp16=float(row['FP16']);int8=float(row['INT8'])
        drop=float(row['signed_drop_FP16_minus_INT8']);limit=float(row['unchanged_limit'])
        equal(fp16-int8,drop)
        expected=.03 if row['model']=='YOLO11s' else (.10 if row['metric'] in ['Crop IoU','Weed IoU'] else .05)
        equal(limit,expected)
        assert drop<=limit and row['result']=='PASS'
        for prec,key in [('FP16','fp16_report'),('INT8','int8_report')]:
            rel=row[key]
            if rel not in reports:reports[rel]=json.loads((ROOT/rel).read_text())
            report=reports[rel]
            assert report['processed_samples']==(1112 if row['model']=='YOLO11s' else 1526)
            if row['model']=='YOLO11s':
                name={'AP50-95':'mAP50_95','AP50':'mAP50','AR@100':'AR100'}[row['metric']]
                equal(row[prec],report['metrics'][name])
            else:
                equal(row[prec],segmentation(report)[row['metric']])
    assert len(reports)==8
    for row in rows('accuracy/coverage_checks.csv'):
        assert int(row['expected_images'])==int(row['processed_images'])
        assert int(row['skipped_images'])==0
        if 'yolo' in row['evaluation']:assert int(row['annotations'])==1886
    classes=rows('accuracy/yolo_per_class_AP.csv')
    assert len(classes)==24
    for row in classes:equal(float(row['fp16_AP50_95'])-float(row['int8_AP50_95']),row['signed_drop_AP50_95'])
    energy=rows('energy/Table8_windows.csv')
    assert len(energy)==6
    for row in energy:
        energy_wh=float(row['table_energy_unrounded_Wh'])
        equal(energy_wh*3600/int(row['recorded_inferences']),row['table_J_per_inference'])
        if row['board']=='Nano' and row['precision']=='FP16':
            equal(float(row['mean_logger_power_W'])*float(row['runtime_s'])/3600,energy_wh)
            equal(energy_wh,61.330743,abs_tol=5e-7)
            equal(row['full_logger_energy_Wh'],62.051665,abs_tol=5e-7)
        else:equal(energy_wh,row['full_logger_energy_Wh'])
    print(f'PASS: {hashes} file hashes; 14 accuracy comparisons; 8 accuracy reports;')
    print('4 segmentation confusion matrices; 24 class AP differences; 6 energy rows.')
    print('This checks supplied records, not new inference, COCO AP scoring, raw power integration,')
    print('or historical-plan internal precision. Run verify_revision_additions.py for recovered model diagnostics.')

if __name__=='__main__':main()
