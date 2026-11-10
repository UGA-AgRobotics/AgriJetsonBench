# Detection inference split policy

- Final reported Jetson detection benchmarking must use the frozen `test` split.
- Quick parser checks, ONNX sanity checks, and first device-side debugging should use the fixed validation pilot subset.
- Do not tune or debug repeatedly on the test split.
