"""Export the diagnostic plan's metadata without an execution context."""
import hashlib
import json
from pathlib import Path

import tensorrt as trt


run = Path(__file__).resolve().parent / 'run'
provenance = json.loads((run / 'provenance.json').read_text())
assert provenance['build_exit_code'] == 0
assert provenance['all_originals_hash_match']
plan = Path(provenance['diagnostic_plan']['path']).read_bytes()
assert hashlib.sha256(plan).hexdigest() == provenance['diagnostic_plan']['sha256']

logger = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(logger, '')
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(plan)
assert engine is not None, 'Diagnostic engine could not be deserialized'
assert engine.profiling_verbosity == trt.ProfilingVerbosity.DETAILED
inspector = engine.create_engine_inspector()
raw = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
data = json.loads(raw)
assert len(data['Layers']) == engine.num_layers
assert all(isinstance(row, dict) for row in data['Layers'])
(run / 'detailed_inspector.json').write_text(raw + '\n')

bindings = []
for index in range(engine.num_io_tensors):
    name = engine.get_tensor_name(index)
    bindings.append({
        'name': name,
        'mode': str(engine.get_tensor_mode(name)),
        'dimensions': list(engine.get_tensor_shape(name)),
        'datatype': str(engine.get_tensor_dtype(name)),
        'format': str(engine.get_tensor_format(name)),
        'format_description': engine.get_tensor_format_desc(name),
        'vectorized_dimension': engine.get_tensor_vectorized_dim(name),
    })
assert next(row for row in bindings if row['mode'] == 'TensorIOMode.INPUT')['dimensions'] == [1, 3, 640, 640]
result = {
    'diagnostic_plan_sha256': provenance['diagnostic_plan']['sha256'],
    'tensorrt_runtime': trt.__version__,
    'profiling_verbosity': str(engine.profiling_verbosity),
    'inspector_entries': engine.num_layers,
    'io_tensors': bindings,
    'execution_context_created': False,
    'inference_executed': False,
}
(run / 'engine_io_and_inspection.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
