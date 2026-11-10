Supplementary evidence for ATECH-D-26-03953
Updated 10 November 2026

MAIN RESULTS
accuracy/accuracy_comparison_both_boards.csv and energy/Table8_windows.csv
correspond to the revised manuscript. Supplementary Material S1 explains the
scope and definitions. Four new AGX reference accuracy evaluations and four
retained Nano evaluations use common coverage and scoring: 1,112 YOLO images
with 1,886 annotations and 1,526 SegFormer pairs scored against original masks.
The 14 comparisons use the unchanged limits. Native AGX FP16 accuracy and
historical reference evaluations remain separately labelled.

COMMENT 4
fallback/diagnostic_AGX/ and fallback/diagnostic_Nano/ contain separate
cache-only rebuild reports, detailed Inspector exports, tactic evidence, code
or patches, and provenance. Original plans are unchanged. Their internal
precision maps remain unverified. Diagnostic tactics are not assignments of
the historical engines and are not used to replace accuracy or energy results.
Nano's diagnostic used 25 W; its historical battery run used 15 W.

RECOVERED RECORDS
accuracy/recovered_AGX_predictions/ contains the two original aligned YOLO
prediction JSONs with verified hashes. Matching COCO annotations and a compatible
COCO evaluator are additionally required to recompute AP.
diagnostics/recovery_addendum/ contains the four observed/predicted model rows,
historical calculation input and a portable numerical checker. It also contains
the recorded AGX SegFormer compilation command/log; no executable is bundled.
Historical recovery notes describe availability at their original dates.
provenance/RELEASE_UPDATE_20261110.txt gives the current status.

ENERGY AND LIMITATIONS
Logger and runtime windows differ. The Nano 25 W FP16 energy row uses the
mean-power/runtime estimate, not its full-logger integral. The displayed 20%
stop condition is confirmed; the fixed 5% text in old Nano logs is not a charge
measurement. Calibration/test overlap, SegFormer preprocessing mismatch and
historical precision/clock limitations remain documented in S1 and limitations/.

VERIFICATION - PYTHON 3 STANDARD LIBRARY
    python3 verify_revision_additions.py
This checks the release file hashes, reported accuracy/energy arithmetic,
segmentation confusion matrices, recovered diagnostic input calculations,
prediction-file identity, and diagnostic export summaries. It does not perform
new inference, COCO AP evaluation, or independent raw power-trace integration.

SCOPE
Original scientific files are retained byte-for-byte; newly derived documentation
is separate. Original machine paths are provenance, not bundled engine paths.
Engine binaries, trained weights, raw image/mask datasets, full logger archives
and CUDA/TensorRT dependencies are not bundled in this compact supplement.
Large AGX recovery archives are indexed separately. Full historical calibration
consumption and the original final Table 8 generator remain unresolved.
This is prepared review material, not a claim of public repository availability.
Existing third-party rights and dataset/software licence restrictions apply;
no new blanket licence is asserted. No credentials or journal correspondence
are included in this evidence archive.
