"""Build once in a new scratch directory, recording provenance and integrity."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

benchmark = Path('/home/nanohasan/Benchmark')
confirm = benchmark / 'resolution_ablation/nano_super_15w_precision_confirmation'
meta_path = confirm / 'segformerb0/s640/engines/segformerb0_s640_int8_meta.json'
meta = json.loads(meta_path.read_text())
package = benchmark / 'Jetson Orin Nano Super Developer Kit/Jetson_benchmark_segformerb0_resab'
battery = benchmark / ('battery/inference_pure_engine/battery_inference_pure_engine_runs/nano_super/'
                       'nano_super_pure_engine_segformer_b0_int8_640_20260508T201856Z')
config = json.loads((battery / 'run_config.json').read_text())
assert config['selected_config']['engine_path'] == meta['engine_path']
assert meta['input_dims'] == [1, 3, 640, 640] and meta['batch_size'] == 1
assert meta['workspace_mb'] == 4096 and meta['allow_fp16_fallback']
originals = {
    'historical_plan': Path(meta['engine_path']),
    'original_onnx': Path(meta['source_onnx_path']),
    'historical_builder_onnx': Path(meta['build_onnx_path']),
    'original_calibration_cache': Path(meta['cache_path']),
    'original_builder_source': package / 'jetson/build_int8_calibration_cache.cpp',
    'original_builder_wrapper': package / 'scripts/build_int8_cache_and_engine.sh',
    'historical_build_metadata': meta_path,
    'historical_build_log': Path(meta['build_log_path']),
    'battery_run_config': battery / 'run_config.json',
}

def digest(path):
    sha256 = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            sha256.update(block)
    return sha256.hexdigest()

def stamp():
    return datetime.now(timezone.utc).isoformat()

root = Path(__file__).resolve().parent
run = root / 'run'
run.mkdir()
before = {name: digest(path) for name, path in originals.items()}
for label, key in [
    ('historical_plan', 'sha256_engine'), ('original_onnx', 'sha256_onnx'),
    ('historical_builder_onnx', 'sha256_build_onnx'), ('original_calibration_cache', 'sha256_cache'),
    ('historical_build_log', 'sha256_build_log'),
]:
    assert before[label] == meta[key], (label, 'does not match historical metadata')
assert before['original_onnx'] == before['historical_builder_onnx']
cache = run / 'copied_calibration.cache'
shutil.copyfile(originals['original_calibration_cache'], cache)
cache.chmod(0o444)
assert digest(cache) == before['original_calibration_cache']
with cache.open('rb') as file:
    assert file.readline().rstrip() == b'TRT-100300-EntropyCalibration2'
provenance = {'started_utc': stamp(), 'recorded_historical_metadata_version': meta['tensorrt_version'],
              'originals': {k: {'path': str(v), 'before_sha256': before[k]} for k, v in originals.items()},
              'copied_cache_before_sha256': digest(cache),
              'scope': 'Separate diagnostic build and metadata-only inspection',
              'battery_mode': '15 W', 'diagnostic_input_shape': [1, 3, 640, 640],
              'workspace_mib': 4096, 'int8': True, 'fp16_fallback_enabled': True,
              'profiling_verbosity': 'kDETAILED', 'tf32_override_environment': os.getenv('NVIDIA_TF32_OVERRIDE')}
record = run / 'provenance.json'
record.write_text(json.dumps(provenance, indent=2) + '\n')
with (run / 'full_build.log').open('w') as log:
    def command(args, *, check=True):
        print('$ ' + shlex.join([str(x) for x in args]), file=log, flush=True)
        return subprocess.run(args, cwd=run, stdout=log, stderr=subprocess.STDOUT, check=check)

    try:
        version_command = [
            '/usr/src/tensorrt/bin/trtexec', f"--loadEngine={originals['historical_plan']}",
            '--getPlanVersionOnly', '--skipInference',
        ]
        version = subprocess.check_output(version_command, stderr=subprocess.STDOUT, text=True)
        (run / 'historical_plan_version.log').write_text(version)
        assert 'Plan was created with TensorRT version 10.3.0.30' in version
        provenance['historical_plan_serialization_version'] = '10.3.0.30'
        command(['dpkg-query', '-W', 'libnvinfer10', 'libnvinfer-dev', 'libnvonnxparsers10'])
        mode = subprocess.check_output(['nvpmodel', '-q'], text=True)
        provenance['diagnostic_nvpmodel'] = mode.strip()
        print(mode, file=log, flush=True)
        binary = run / 'cache_only_builder'
        plan = run / 'segformerb0_nano_diagnostic_only.engine'
        command(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-I/usr/local/cuda/include',
                 str(root / 'cache_only_builder.cpp'), '-o', str(binary), '-lnvinfer', '-lnvonnxparser'])
        command(['ldd', str(binary)])
        result = command([str(binary), str(originals['historical_builder_onnx']), str(cache), str(plan)], check=False)
        provenance['build_exit_code'] = result.returncode
        if result.returncode == 0:
            provenance['diagnostic_plan'] = {'path': str(plan), 'sha256': digest(plan), 'size_bytes': plan.stat().st_size}
        sys.exit(result.returncode)
    finally:
        after = {name: digest(path) for name, path in originals.items()}
        for label, value in after.items():
            provenance['originals'][label]['after_sha256'] = value
        provenance['all_originals_hash_match'] = before == after
        provenance['copied_cache_after_sha256'] = digest(cache)
        provenance['finished_utc'] = stamp()
        record.write_text(json.dumps(provenance, indent=2) + '\n')
        assert before == after and digest(cache) == before['original_calibration_cache']
