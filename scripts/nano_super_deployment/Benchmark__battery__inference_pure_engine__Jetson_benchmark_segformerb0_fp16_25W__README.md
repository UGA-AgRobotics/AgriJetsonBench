# SegFormer-B0 FP16 640 Native 25W Pure Engine Battery Run

This folder is an isolated copy of the pure-engine runner setup for:

- board: Nano Super
- power mode: 25W native
- model: SegFormer-B0
- precision: FP16
- input size: 640
- batch size: 1
- boundary: pure TensorRT engine only, trtexec random input buffers

The measured path does not include JPEG/image loading, preprocessing, postprocessing, NMS, visualization, or prediction saving.

Run order:

```bash
cd ${NANO_HOME}/Benchmark/battery/inference_pure_engine/Jetson_benchmark_segformerb0_fp16_25W
# Privileged device-control steps must be reviewed and run manually on the target system.
bash scripts/preflight_check.sh
bash scripts/smoke_test_no_noise_guard.sh       # optional
bash scripts/run_final_foreground.sh
```

Final outputs are saved under:

```text
${NANO_HOME}/Benchmark/battery/inference_pure_engine/Jetson_benchmark_segformerb0_fp16_25W/battery_inference_pure_engine_runs/nano_super
```
