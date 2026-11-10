# AGX SegFormer B0 FP16 640 Battery Run In Fifty Watt Mode

This folder is an isolated copy of the AGX battery inference runner for the SegFormer B0 FP16 640 run in AGX fifty watt mode.

All new outputs are written under:

```text
${AGRIBENCH_ROOT}/battery/for_agx/runs/agx
```

The FP16 TensorRT engine used by this run is:

```text
${AGRIBENCH_ROOT}/battery/for_agx/model/segformer_b0_fp16_640.engine
```

The copied runner code is under:

```text
${AGRIBENCH_ROOT}/battery/for_agx/code
```

Before a real endurance run, verify the logger reads a real AGX load. The temperature channel can be valid even when INA260 voltage and power are not in the AGX current path.

Run the full endurance experiment with:

```bash
cd ${AGRIBENCH_ROOT}/battery/for_agx/code
scripts/run_battery_inference_agx_with_sudo.sh \
  --config config/battery_inference_agx_segformer_fp16.yaml \
  --config-name segformer_b0_fp16_640 \
  --precision FP16 \
  --mode endurance \
  --set-nvpmodel \
  --status-interval 60
```

Stop the run with Ctrl C at the chosen power bank remaining percentage. The runner finalizes CSV JSON Markdown logs and external logger archive automatically.

For a non endurance preflight preview:

```bash
cd ${AGRIBENCH_ROOT}/battery/for_agx/code
python3 scripts/run_battery_inference_agx.py \
  --config config/battery_inference_agx_segformer_fp16.yaml \
  --config-name segformer_b0_fp16_640 \
  --precision FP16 \
  --mode endurance \
  --dry-run \
  --logger-control none \
  --allow-power-mode-mismatch \
  --no-wait-for-enter \
  --no-sudo-prompt-once
```

After a completed run, regenerate the local summary table with:

```bash
cd ${AGRIBENCH_ROOT}/battery/for_agx/code
python3 scripts/summarize_battery_inference_agx.py \
  --runs-dir ${AGRIBENCH_ROOT}/battery/for_agx/runs/agx \
  --out-dir ${AGRIBENCH_ROOT}/battery/for_agx/summary/agx
```
