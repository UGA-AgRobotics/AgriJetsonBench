"""Classify named tactic families separately from tensor formats."""
from collections import Counter
import csv
import json
from pathlib import Path
import re


FAMILY_TOKENS = {
    'FP16': {'f16f16', 'f16f32', 'fp16fp16', 'fp16', 'half', 'hgemm', 'h16816gemm'},
    'INT8': {'i8i8', 'i8i32', 's8s8', 's8s32', 'int8', 'int8int8'},
    'TF32': {'tf32', 'tf32tf32', 'tf32f32', 'tfloat32'},
}
DATATYPE_TOKEN = re.compile(r'(?:tfloat32|tf32|fp16|f16|int8|i8|s8|fp32|f32|i32|s32|bf16|f64|u8|half){1,4}')


def tactic_evidence(tactic):
    tokens = set(re.findall(r'[a-z0-9]+', tactic.lower()))
    datatype_tokens = sorted(token for token in tokens if DATATYPE_TOKEN.fullmatch(token))
    evidence = {family: sorted(tokens & markers) for family, markers in FAMILY_TOKENS.items()
                if tokens & markers}
    if {'depthwisehmma', 'fp16nhwcx8'} <= tokens:
        evidence.setdefault('FP16', []).append('depthwiseHMMA + FP16NHWCx8')
        datatype_tokens.append('fp16nhwcx8')
    if not evidence:
        float32_pattern = re.search(r'(?:^|_)f32f32_(?:f32f32|f32)_f32(?:_|$)', tactic.lower())
        if float32_pattern:
            evidence['FP32'] = [float32_pattern.group().strip('_')]
    if len(evidence) == 1:
        family = next(iter(evidence))
        return family, evidence, datatype_tokens, 'Selected tactic name contains a family marker'
    if evidence:
        return 'unknown', evidence, datatype_tokens, 'Selected tactic name contains conflicting family markers'
    reason = 'Inspector supplies no tactic name' if not tactic else 'Selected tactic name has no recognized family marker'
    return 'unknown', evidence, datatype_tokens, reason


def formats(ports):
    return [{
        'name': port.get('Name', ''),
        'dimensions': port.get('Dimensions'),
        'format_datatype': port.get('Format/Datatype', ''),
    } for port in ports]


def main():
    run = Path(__file__).resolve().parent / 'run'
    inspector = json.loads((run / 'detailed_inspector.json').read_text())
    counts = Counter()
    unknown_reasons = Counter()
    tactic_names = {family: set() for family in [*FAMILY_TOKENS, 'FP32']}
    explicit_precision_keys = Counter()
    rows = []
    for index, layer in enumerate(inspector['Layers']):
        tactic = layer.get('TacticName') or ''
        family, evidence, markers, reason = tactic_evidence(tactic)
        counts[family] += 1
        if family == 'unknown':
            unknown_reasons[reason] += 1
        else:
            tactic_names[family].add(tactic)
        reported_precision = {key: value for key, value in layer.items()
                              if 'precision' in key.lower() or 'accumulation' in key.lower()}
        explicit_precision_keys.update(reported_precision.keys())
        rows.append({
            'inspector_index_zero_based': index,
            'name': layer.get('Name', ''),
            'layer_type': layer.get('LayerType', ''),
            'parameter_type': layer.get('ParameterType', ''),
            'selected_tactic_name': tactic,
            'selected_tactic_value': layer.get('TacticValue', ''),
            'tactic_family': family,
            'classification_basis': reason,
            'family_evidence_tokens': json.dumps(evidence, sort_keys=True),
            'all_tactic_datatype_tokens': ' '.join(markers),
            'inspector_precision_fields': json.dumps(reported_precision, sort_keys=True),
            'input_tensor_formats': json.dumps(formats(layer.get('Inputs', []))),
            'output_tensor_formats': json.dumps(formats(layer.get('Outputs', []))),
        })
    assert rows
    assert sum(counts.values()) == len(rows)
    with (run / 'layer_precision.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'inspector_entries': len(rows),
        'counts_by_selected_tactic_family': {key: counts[key] for key in ['FP16', 'INT8', 'TF32', 'FP32', 'unknown']},
        'distinct_names_by_family': {key: len(names) for key, names in tactic_names.items()},
        'unknown_reasons': dict(unknown_reasons),
        'entries_with_named_tactic': sum(bool(row['selected_tactic_name']) for row in rows),
        'explicit_precision_field_names_and_counts': dict(explicit_precision_keys),
        'classification_rules': {
            'standalone_tokens': {key: sorted(tokens) for key, tokens in FAMILY_TOKENS.items()},
            'additional_FP16_rule': 'Both depthwiseHMMA and FP16NHWCx8 occur in the selected tactic name',
            'additional_FP32_rule': 'Explicit f32f32_f32_f32 or f32f32_f32f32_f32 name sequence, in the absence of another family marker',
        },
        'count_unit': 'Inspector entries classified from selected tactic names. Fused entries may represent multiple ONNX operations.',
        'tensor_format_role': 'Recorded per input and output. Never used to classify tactic family.',
        'accumulation_role': 'Raw naming markers retained. An explicit Inspector field is needed for a separately reported accumulation precision.',
    }
    (run / 'tactic_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
