# Inference Split Policy

## Final reported benchmark split

- Use the frozen `test` split for final Jetson-side reporting.

## Validation usage

- Use the locked `val` split for training-time evaluation and qualitative checks.

## Important rule

- Do not tune hyperparameters on the frozen `test` split.
