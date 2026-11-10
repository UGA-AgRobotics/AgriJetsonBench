# AgriJetsonBench missing-material recovery

Search completed on Sapelo2 on 4 November 2026. This is a small recovery addendum; the prepared Zenodo release and original project files were not changed.

Recovered item C: the original six-row `paper_tables/energy_model_validation.csv` contains the complete four observed/predicted rows. Select data rows 1–4, identified by `eta_fit_source=leave_one_out_same_board_mode`. Columns `observed_avg_external_power_w` / `predicted_avg_external_power_w` and `observed_j_per_inference` / `predicted_j_per_inference` contain the two complete vector pairs. The final two rows are same-row calibration cases, not held-out observations.

The original CSV was already retained; its input relationship is now confirmed. Its bytes and existing checksum are preserved. The newly recovered calculation code is the exact Python standard input from a successful saved execution on 2 November at 14:40:52.358 UTC. It is preserved under `verified_recovery/historical_execution/20261102T144052Z/standard_input.py`; this filename is an extraction destination, not a claim that a standalone script originally existed. The provenance record identifies its original location. The historical code retains its original absolute paths and audit-output statements and was not executed during this recovery.

Independent arithmetic in the original Python 3.12.7 environment reproduces both saved four-row diagnostics exactly: power MAE 7.220501 W, RMSE 8.159767135901765 W, MAPE 57.25376920517581%, centered R² −19.451593222725112; energy MAE 0.098442 J, RMSE 0.09942322126897721 J, MAPE 56.694102967388915%, centered R² 0.5981947118159143. See `recovery_verification.json`.

Still not found in the searched locations:

- A: both specified November 3 AGX YOLO prediction files. No candidate with either required size or hash was found; older outputs were not substituted.
- B: `project_progress_agx.zip`, `project_progress_nano.zip`, and `agx_selected_internal_telemetry_main_runs.zip`, named by the agreement workflow. No substitute archives were created.
- D: the historical final Table 8 generator. The recovered diagnostics code contains later arithmetic checks; the later manuscript assembly script writes `Table8_windows.csv`. Neither establishes the requested generator.
- E: a frozen per-image historical calibration consumption record. `calibration_record_index.csv` identifies four already retained metadata records linking manifest paths, counts, caches and engines. These links do not establish which images were actually consumed or complete historical cache provenance.
- F: a build script/recorded compilation command, or optional executable, for `eval_segmentation_aligned.cpp`. The older evaluator is distinct. The source reported as already included was not recopied.

`recovery_manifest.csv` records every requested item. `search_locations.csv` specifies the bounded project, backup and log searches, including paths that no longer exist. The old release was searched through its existing member index, not rebuilt. AGX/Nano workstation storage was not accessible through this search. Negative findings apply only to the disclosed locations.

`external_assets.csv` lists available original dataset packages, portable image/mask packages, seven models’ retained weights/ONNX files, additional staged copies, and separately labelled workstation engine references. Current local sizes were checked. Existing large-file hashes and licence records are attributed to their sources; those hashes were not newly verified. No datasets or large binaries are in this package. `unverified_candidates/candidate_index.csv` keeps leads separate from verified recovery files.

Original scientific files are byte-identical. Only item C’s numerical reproducibility gap is addressed. Four observations do not establish independent model validation, scientific accuracy or predictive generalization. The upload notes’ other precision, calibration-disjointness, cache-provenance, clock-alignment and energy-window limitations remain. No experiments, compilation, training, inference, calibration, engine builds, upload or publication occurred. No new licence grant is asserted; existing authorship, provenance and licence terms continue to apply.
