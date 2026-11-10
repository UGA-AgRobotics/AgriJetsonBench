#!/usr/bin/env python3
"""Check the supplied diagnostic vectors; no model fitting or inference.

This portable checker was added with the 4 November 2026 recovery addendum.
It is separate from the preserved historical standard-input code.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

root = Path(__file__).resolve().parent
inputs = list((root / 'sapelo2_recovery/verified_recovery').rglob('energy_model_validation.csv'))
assert len(inputs) == 1, 'Expected one diagnostic input CSV'
input_file = inputs[0]
expected = json.loads((root / 'sapelo2_recovery/recovery_verification.json').read_text())
assert hashlib.sha256(input_file.read_bytes()).hexdigest() == expected['recovered_input_sha256']
with input_file.open(newline='') as stream:
    rows = [row for row in csv.DictReader(stream)
            if row['eta_fit_source'] == 'leave_one_out_same_board_mode']
assert len(rows) == 4
assert [row['run_id'] for row in rows] == expected['selected_run_ids']

def stats(observed, predicted):
    y = [float(row[observed]) for row in rows]
    p = [float(row[predicted]) for row in rows]
    errors = [a - b for a, b in zip(p, y)]
    n = len(y)
    mean = sum(y) / n
    sst = sum((v - mean) ** 2 for v in y)
    sse = sum(e * e for e in errors)
    return {'n': n, 'MAE': sum(abs(e) for e in errors) / n,
            'RMSE': math.sqrt(sse / n),
            'MAPE_percent': sum(abs(e) / abs(v) for e, v in zip(errors, y)) * 100 / n,
            'R2_centered': 1 - sse / sst}

result = {'scope': 'Recalculation of error statistics from four stored prediction/reference rows; no model fitting.',
          'input_sha256': hashlib.sha256(input_file.read_bytes()).hexdigest(),
          'selected_run_ids': expected['selected_run_ids']}
for key, observed, predicted in [
    ('power_LOO_four_rows', 'observed_avg_external_power_w', 'predicted_avg_external_power_w'),
    ('energy_LOO_four_rows', 'observed_j_per_inference', 'predicted_j_per_inference')]:
    actual = stats(observed, predicted)
    target = expected[key]['values']
    assert all(math.isclose(actual[k], target[k], rel_tol=1e-12, abs_tol=1e-12) for k in target), key
    result[key] = {'values': actual, 'matches_saved_summary': True,
                   'exact_equality_in_this_run': actual == target}
print(json.dumps(result, indent=2))
