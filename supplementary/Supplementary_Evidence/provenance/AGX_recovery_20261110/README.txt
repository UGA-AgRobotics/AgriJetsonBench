AgriJetsonBench AGX file recovery
ATECH-D-26-03953 | 2026-11-10

Both YOLO11s detection files were copied byte-for-byte from the original AGX
run directory. Their byte counts and SHA-256 digests match A_fp16 and A_int8
in the supplied recovery manifest. SHA256SUMS.txt checks the packaged payloads.

The manifest records two large archives in place. Both passed ZIP integrity
checks. The telemetry archive also matches its recorded historical SHA-256.
The SHA-256 for project_progress_agx.zip was measured on the AGX copy.

Still missing in the scoped search are project_progress_nano.zip, the source
that generated the submitted six-row Table 8, and an original per-image record
of calibration consumption. The separate candidate index records related files
and why they were excluded.

Search scope included the AGX Benchmark tree, project progress and selection
folders, project backups, GitHub and Zenodo staging copies, Downloads, the
revision workspace, and the related evidence folders. Archive member names
and selected code, reports, and build logs were checked. Table 8 searches
included Table8, table8, 61.330743, and 61.331.

Original files were preserved. No predictions, tables, or calibration lists
were recreated. No experiments ran and the recovered diagnostics and recorded
compilation command were left outside this package.

Expected YOLO size and SHA-256 source
/home/agxhasan/Downloads/AgriJetsonBench_Recovery_Addendum_20261104.zip
AgriJetsonBench_Recovery_Addendum_20261104/sapelo2_recovery/recovery_manifest.csv
Manifest SHA-256 0a3e9603f8d8bf17d58163693dc19a09418bc985f8c63764f8c86008f9a0e313
