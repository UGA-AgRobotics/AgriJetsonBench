# Inference split policy

## Final reported Jetson benchmark later
Use the frozen `test_full.txt` manifest, which corresponds to the fixed 20% test split.

## Quick board-side checks later
Use `val_jetson_pilot.txt` for:
- ONNX parser sanity checks
- first TensorRT conversion tests
- first latency loop tests
- wiring and logging checks

## Rule
Do not tune hyperparameters on the test split.
