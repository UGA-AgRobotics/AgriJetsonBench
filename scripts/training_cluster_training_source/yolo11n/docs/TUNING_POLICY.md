# Tuning policy for this project

## Main rule
Do not tune on the test split.

## Recommended scope
For this benchmark project, keep tuning controlled and limited.

A good policy is:
- use tuning only for `YOLO11n` at this stage
- use the frozen train + val setup
- keep the search budget modest
- freeze the final recipe after tuning

## Why not tune everything heavily?
Because the paper is mainly about:
- reproducible benchmarking
- deployment behavior on Jetson
- accuracy-latency-energy trade-offs

If every model gets a large custom tuning effort, the project can become too large and less fair.

## What this package does
- optional Optuna stage
- tuning uses only train + val
- final training can reuse the best hyperparameters automatically
- the final Jetson benchmark later still uses the frozen test split
