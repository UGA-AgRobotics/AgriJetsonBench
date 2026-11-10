# Reviewer Comment 4 · SegFormer-B0 INT8 on Jetson AGX Orin

**The original plan exposes FP32 input and output bindings. Its internal INT8, FP16 and FP32 execution assignments remain unverified.** The original build enabled calibrated INT8 and permitted FP16. The saved engine contains layer names only.

## Engine identity

Submitted Table 8's **AGX Orin 15 W SegFormer-B0 INT8** row matches run `agx_battery_20260507T111149Z` (640 × 640, batch 1, 414,130 inferences). The saved run config names this engine, and all 200 saved battery chunk summaries record the same measured engine path and SHA-256. The archived model-artifact copy has the same hash.

- Original path `/home/agxhasan/Benchmark/completed_Inferences/segformerb0/model/best_int8.engine`
- Size 9,761,436 bytes
- SHA-256 `2290bdfd45e4be89634761d075b756604e301275329aeb753253a1af5caa7245`
- Serialized TensorRT version **10.4.0.26** from the saved plan-version check. The fresh inspection loaded the local TensorRT 10.4 libraries and returned runtime version 10.4.0.

## What the original engine exposes

A fresh C++ Engine Inspector check deserialized the original bytes without creating an execution context. `getProfilingVerbosity()` returned `kLAYER_NAMES_ONLY`. The 187 entries in `original_engine_inspector.json` are strings in the same order as the earlier `trtexec` export. There are no structured layer operation types, layer input/output formats, selected tactics, compute precision or accumulation precision fields. TensorRT emitted a device-model portability warning during deserialization and returned the metadata successfully. [NVIDIA Engine Inspector documentation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html)

| Assignment or setting | Finding | Evidence |
|---|---|---|
| External input `images` | **Confirmed FP32**, row-major linear, 1 × 3 × 640 × 640 | Plan API `kFLOAT`, `kLINEAR` |
| External output `logits` | **Confirmed FP32**, row-major linear, 1 × 3 × 640 × 640 | Plan API `kFLOAT`, `kLINEAR` |
| INT8 enabled and FP16 permitted | **Confirmed build settings** | Saved metadata and C++ builder flags |
| Internal INT8 execution | **Unconfirmed per layer** | Saved scales and INT8 eligibility do not identify selected kernels |
| Internal FP16 or FP32 execution or fallback | **Unconfirmed per layer** | No final precision assignments in the inspected plan or saved build log |

Names suggest patch-embedding convolutions, fused attention and normalization-related entries, MLP/depthwise convolutions, decoder convolutions and resizing, and reformat/copy entries. Examples are `/model/encoder/patch_embeds.0/proj/Conv`, `_gemm_mha_v2_myl7_4`, and `/model/decode_head/Resize_3`. These are name-based descriptions. The input and output binding types establish the engine interface only.

## What the build records establish

- **Build log.** `best_int8_build.log` contains 122 missing-scale/zero-point warning lines. For example, lines 3 and 5 name `model.encoder.patch_embeds.0.norm.weight_output` and `ONNXTRT_Broadcast_output` and say to expect a non-INT8 implementation for layers using those tensors. The warning count is a count of log lines about tensors. The log provides no final layer assignments and gives no FP16/FP32 split. See `build_log_excerpts.txt`.
- **ONNX.** The retained build input `best_int8_trt.onnx` passes ONNX validation and is byte-identical to `best.onnx`. The opset 17 graph has 618 nodes, including Conv, MatMul, LayerNormalization, Softmax and Resize. The graph has FP32 I/O and 188 FP32 initializers, with no QuantizeLinear or DequantizeLinear nodes. It describes the source graph for implicit INT8 calibration.
- **Calibration cache.** The retained `best_int8_calibration.cache` has the `TRT-100400-EntropyCalibration2` header and 641 named scale records. The build metadata gives the test-manifest path and a 256-image cap and count. A frozen list of images consumed and a build-time cache digest were unavailable. Scale records describe quantization ranges.
- **Build configuration.** The saved C++ helper sets INT8 and conditionally enables FP16, with no explicit layer precision constraints or profiling-verbosity override. The metadata records 640 × 640, batch 1, 4096 MiB workspace, FP16 allowed and the local 10.4.0.26 stack. The scoped search of project copies, battery records and archives found no matching detailed Inspector export, tactic-selection log or saved tactic timing cache.

For Comment 4, describe the reported accuracy and battery results as measurements of the **calibrated INT8-enabled TensorRT engine with FP16 permitted and FP32 I/O**. Its saved records leave the internal precision split and the effect of any specific fallback operation unresolved.

## Diagnostic rebuild feasibility

**Technically possible with the retained assets, pending cache acceptance by the builder.** The matching ONNX, calibration cache, C++ builder source and known build settings are available. The local TensorRT 10.4.0.26 stack includes the runtime, parser and builder resources. Its matching parser header is in an existing local development package. Paths and hashes are in `source_index.csv`.

If a diagnostic build is warranted, use a separate directory, the retained ONNX and a copy of the cache, and the original helper settings. Set `ProfilingVerbosity::kDETAILED` before building and record whether the cache is accepted without requesting calibration batches. Stop if it requests batches. Save the diagnostic plan and inspect it without running inference. Separate per-layer tensor formats from any compute precision actually reported. Tactics and fusions may differ in a new build, so report its assignments under the diagnostic plan's own hash. For the historical plan, retain the limitation above. [NVIDIA TensorRT precision and tactic selection](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)

The original engine SHA-256 matched before and after inspection. This investigation used saved records and metadata-only inspection. No engine rebuild, calibration, inference, accuracy evaluation or hardware benchmark was performed.
