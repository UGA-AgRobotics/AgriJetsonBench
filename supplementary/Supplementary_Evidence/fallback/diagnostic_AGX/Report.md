# Reviewer Comment 4 · SegFormer-B0 diagnostic TensorRT build

The separate AGX Orin build succeeded using the retained calibration cache. The diagnostic plan shows a mix of INT8 and floating-point tactics. Its assignments apply to the diagnostic plan. The historical battery engine still exposes layer names only.

## Build and preservation

The diagnostic build used the same ONNX, TensorRT 10.4.0.26, 640 × 640 input, batch 1, 4096 MiB workspace, INT8 enabled, and FP16 permitted. The original C++ builder was copied and patched to set `kDETAILED`, retain verbose logs, and enforce cache-only calibration. It had an empty dataset directory and would exit immediately if TensorRT requested a batch or a cache write.

TensorRT read the 34,723-byte cache copy twice and logged that it generated scales using the cache. The build completed with **zero calibration batches and zero cache writes**. The original engine, ONNX, cache, and builder hashes matched before and after. The copied cache remained byte-identical to the original. TensorRT's normal build-time tactic selection ran. Inspection deserialized the plan without an execution context. There was no application inference, accuracy evaluation, power measurement, or performance benchmark.

- Original plan SHA-256 `2290bdfd45e4be89634761d075b756604e301275329aeb753253a1af5caa7245`
- Diagnostic plan SHA-256 `7520ee1eae1fdcf9ab15f75be6da6eede8d23727aa03ae794c78735c2603e27a`
- Retained ONNX SHA-256 `d2a4c21f7c8baadfa40cac7f951b90d923ff6901e15d9e1defb3a5987f77e6cd`
- Retained and copied cache SHA-256 `1e659fff8936e34ffdb8d2ab60743138821f6110b35b190c63723431221b4e40`

## Original and diagnostic metadata

| | Historical battery plan | Diagnostic plan |
|---|---|---|
| Plan size | 9,761,436 bytes | 9,847,324 bytes |
| Serialized TensorRT version | 10.4.0.26 | 10.4.0.26 |
| Engine bindings | `images` and `logits`, FP32 linear, 1 × 3 × 640 × 640 each | Same |
| Profiling verbosity | `kLAYER_NAMES_ONLY` | `kDETAILED` |
| Inspector entries | 187 layer-name strings | 193 records with layer types, tensors and available tactics |

The exports share 90 exact name occurrences when counted with multiplicity. Fused and generated layer names can change between builds. The names alone cannot align every operation or establish identical tactics even for an exact name match.

## Precision evidence from the diagnostic plan

The Inspector reports the format/datatype of layer inputs and outputs, plus selected tactic names or IDs where available. There are 450 Half, 134 Float, and 17 Int8 tensor appearances across the listed inputs and outputs. These counts include repeated tensors and reformatting. They measure tensor formats. The export contains no dedicated execution-precision or accumulation-precision field. [NVIDIA Engine Inspector documentation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/engine-tools.html)

The CSV preserves the selected tactic names and separates their datatype markers from tensor and weight datatypes. Among the selected XMMA tactic names with identifiable datatype groups, **52 entries have FP16-family markers, five have INT8-family markers, and one has a TF32-family marker**. These groups use INT8 first when an integer tactic also contains floating-point markers. Opaque tactic names and entries without a named tactic are left unclassified.

| Example diagnostic entry | Tensor formats | Selected tactic evidence |
|---|---|---|
| `/model/encoder/patch_embeds.0/proj/Conv` | Half → Half, Half weights | `f16f16_f16f16_f16` FP16-family convolution |
| `/model/encoder/blocks_0_0/attn/q/MatMul_myl7_1` | Half and Float inputs → Half | `f16f16_f16f16_f16` FP16-family GEMM in the attention projection |
| `/model/encoder/blocks.1.0/attn/sr/Conv` | Int8 → Half, Int8 weights | `i8f16_i8i32_f32` INT8-family convolution with a Half output |
| `/model/encoder/blocks_1_0/mlp/fc2/MatMul_myl27_2` | Float → Float | `f32f32_tf32f32_f32` TF32-family GEMM despite FP32 tensor formats |
| Decode-head fused convolution and ReLU | Int8 → Int8, Int8 weights | `i8i8_i32_f32` INT8-family convolution |
| `_gemm_mha_v2_myl7_4` and normalization-related fused entries | Half or mixed Half/Float, depending on entry | Opaque tactic names leave the internal arithmetic unresolved |

The `i32` and `f32` tactic markers are preserved as reported. The export does not separately identify the accumulation precision of each operation. In particular, a Half result from an INT8 convolution and a Float result from a TF32 GEMM show why output datatype alone cannot classify arithmetic.

## Conclusion for Comment 4

The diagnostic plan supplies concrete examples of FP16, INT8 and TF32 tactic selection using the saved calibration scales. For the historical engine, the defensible statement remains that TensorRT built it with calibrated INT8 enabled, FP16 permitted, and FP32 input/output bindings. Its layer-level precision and the effect of any particular fallback operation on the reported results remain unresolved.

The full build log, raw Inspector JSON, per-entry CSV and provenance are in this folder. The diagnostic plan's absolute path is recorded in `diagnostic_engine_sha256.txt`.
