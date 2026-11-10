# AgriJetsonBench

This snapshot accompanies the 10 November 2026 revision of ATECH-D-26-03953.
It contains Jetson benchmark and battery runners, INA260 logger code,
TensorRT build and evaluation source, analysis scripts, configurations and
compact research evidence. The matching Zenodo archive contains the clean
manuscript, S1, figures, raw deployment records and training evidence.

## Verify the supplied results

From the repository root, use Python 3 and a SHA256 checker.

```sh
sha256sum -c SHA256SUMS.txt
python3 scripts/validate_checksums.py --manifest MANIFEST.csv
python3 supplementary/Supplementary_Evidence/verify_revision_additions.py
```

These commands check file identity and stored results. They cover the aligned
accuracy tables, segmentation confusion matrices, six energy rows, recovered
AGX prediction hashes, descriptive-model arithmetic and the two diagnostic
exports. The checks print their results to the terminal.

## Set up a research workflow

The offline checks use the Python standard library. For general analysis,
create a Python environment and install the packages in `requirements.txt`
or use `environment.yml`. Install the device and logger dependencies listed in
the relevant subdirectory before running a hardware workflow. Supply compatible
JetPack, CUDA and TensorRT versions, model artifacts and permitted data.
Use local paths in a working copy of the configuration.

- **Benchmark and battery runs.** Start with
  `scripts/agx_deployment_configs/battery_inference/`. Its YAML and
  `scripts/run_battery_inference_agx.py` describe the AGX runner. Review its
  `--help` output and supply `--config battery_inference_agx.yaml` from that
  directory when setting up a compatible run. The run directory holds
  `summary.json`, timing records and events. The Nano runners and configs
  under `scripts/nano_super_deployment*` include historical flattened paths.
  Resolve these against the original run layout before using the wrappers.
- **INA260 logging.** See each board's
  `logger_metadata/logger_*/logger_firmware_and_scripts/` and
  `environment/requirements_pc.txt` or `requirements_pi.txt` where present.
  The retained `pi_logger/logger_server.py` records external measurements.
  Configure the matching sensor or serial bridge and an output directory.
- **Engine source and accuracy.** Device packages include
  `jetson/build_tensorrt_engine.py` and calibration helpers. The aligned
  YOLO and SegFormer evaluator sources are in the supplement's `code/`.
  Follow its recorded execution and compilation commands and use the actual
  compatible engine and dataset paths. Aligned SegFormer scoring uses
  `--eval-resolution original`. The recovered command is a build record.
- **Energy analysis.** In the AGX benchmark package, the existing
  `analysis/merge_and_summarize.py --run-root PATH` accepts a compatible saved
  run directory and writes merged/summary tables. See
  `scripts/README_agreement_workflow.md` for the separate historical workflow
  and the external input ZIPs it requires.

## Results and scope

The main aligned reference comparison uses 1,112 YOLO11s images with 1,886
annotations and 1,526 SegFormer-B0 image-mask pairs per board. Earlier AGX
evaluations and the distinct native AGX FP16 accuracy result are stored
separately. The diagnostic exports describe separate plans. Nano diagnostics
used TensorRT 10.3.0.30 in active 25 W mode. The related historical battery
run used 15 W. See the evidence reports for the original metadata discrepancy,
calibration overlap, preprocessing differences and energy-window limits.

The repository supplies split lists, engine identities, configuration files and
compact scientific records. Obtain the original image/mask datasets, model
weights, ONNX/TensorRT binaries and full telemetry separately. Some original
paths are historical records. See the supplement's README, its limitations
and the dataset source notes for details. Existing licences and third-party
notices apply within their stated scope.
