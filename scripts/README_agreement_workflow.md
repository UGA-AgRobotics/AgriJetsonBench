# Historical agreement workflow

`build_agreement_figures_and_decision_map.py` is the original retained analysis
script. It resamples logger/internal telemetry to 1 Hz UTC bins and joins their
overlap. The earlier 76,691 paired samples are preserved in
`historical_pre_revision/paper_tables/`; they are not independent trials.

From this repository root, the original input/output directory layout can be
selected with `--root historical_pre_revision`. The additional required options
are `--agx-project-zip`, `--nano-project-zip` and `--agx-internal-zip`, pointing to
the original curated project_progress_agx.zip, project_progress_nano.zip and
agx_selected_internal_telemetry_main_runs.zip files. Those ZIPs were not present
in the supplied old archive. Related extracted raw records are retained in
RECORD_2, but this script does not accept that tree as a replacement ZIP.

Use a separate copy for any future regeneration. This workflow writes derived
files and indexes; it was not run for this release. Its older window terminology
and descriptive interpretations are superseded by the revised manuscript and
Supplementary Material S1. Current aligned accuracy is independently documented
under `supplementary/Supplementary_Evidence/`.
