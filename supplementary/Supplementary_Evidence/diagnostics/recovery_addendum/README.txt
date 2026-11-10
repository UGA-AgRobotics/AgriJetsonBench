AgriJetsonBench - recovery addendum, 4 November 2026

This small package supplies verified recovery material for incorporation into
the prepared Zenodo release. It is not a replacement for that full release.
The full Zenodo archive, manuscript and S1 have not been modified here.

CONTENTS
sapelo2_recovery/ preserves all ten files from the supplied recovery archive.
Its historical search findings remain unchanged and are superseded only for
the evaluator build-command item by the additional evidence below.

agx_evaluator_build/ contains the recorded compilation command, compilation
log and matching C++ evaluator source recovered from the original uploaded
AGX_Comment3_Aligned_Evaluation.zip. All three files are byte-identical to that
upload and match its original checksum list. The recorded return code is 0.
The command uses historical AGX-specific paths and requires the recorded
TensorRT/CUDA/OpenCV environment. No compilation was performed here and no
compiled executable is bundled.

DIAGNOSTIC CHECK
Run: python3 verify_diagnostics.py
This new portable checker selects the four rows identified by
eta_fit_source=leave_one_out_same_board_mode and recalculates the saved error
statistics. The two native FP16 same-row calibration cases are excluded.
All power and energy metrics reproduce the saved values and agree with
manuscript Table 12 and Supplementary Table S7 at their reported precision.
This verifies arithmetic from stored predictions; it does not regenerate
those predictions, fit the model or establish general predictive validation.

The historical standard_input.py is preserved for provenance. It has absolute
paths and writes an audit output; use the portable checker above for the small
verification task. The original session is not bundled, so its historical
extraction claim has not been independently authenticated here.

REMAINING ITEMS
Both specified November 3 AGX YOLO prediction files; three historical curated
agreement ZIPs; the historical final Table 8 generator; and an original frozen
calibration-consumption list remain unresolved. Existing scientific limitations
are unchanged. Large assets are indexed, not copied or newly hash-verified.

See INTEGRATION_NOTES.txt for the exact documentation updates needed when
these files are incorporated. Original licences and attributions continue to
apply. No experimental, source or historical recovery files were overwritten.
