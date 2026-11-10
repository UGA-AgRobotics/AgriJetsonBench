# Pure Engine Paper Highlight Index

Classification: `KEEP_FOR_PAPER_PURE_ENGINE`

| Model | Task | Precision | Size | Run ID | FPS | P95 ms | Energy Wh | Wh / 1000 samples |
|---|---|---:|---:|---|---:|---:|---:|---:|
| segformer_b0 | segmentation | INT8 | 640 | `nano_super_pure_engine_segformer_b0_int8_640_20260508T201856Z` | 47.176 | 21.5820 | 58.974 | 0.077995 |
| yolo11s | detection | INT8 | 640 | `nano_super_pure_engine_yolo11s_int8_640_20260507T165759Z` | 145.821 | 7.0176 | 58.843 | 0.021318 |

These runs are pure TensorRT engine measurements using trtexec random input buffers. They are not end-to-end image pipeline results.
