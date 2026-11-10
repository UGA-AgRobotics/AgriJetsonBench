# PAPER MARKER SEGFORMER B0 FP16 640 AGX 50W CLEAN RUN

marker: PAPER_CLEAN_RUN_SEGFORMER_B0_FP16_640_AGX_50W_20260508T111833Z

status: clean finalized run

run_id: agx_segformer_b0_fp16_640_20260508T111833Z
run_dir: /home/agxhasan/Benchmark/battery/for_agx/runs/agx/agx_segformer_b0_fp16_640_20260508T111833Z

model: SegFormer B0
task: semantic segmentation
precision: FP16
input_size: 640
batch_size: 1
board: NVIDIA Jetson AGX Orin 64GB Developer Kit
power_mode: MODE_50W id 3
power_source: UGREEN 25000mAh 145W USB C PD power bank
engine: /home/agxhasan/Benchmark/battery/for_agx/model/segformer_b0_fp16_640.engine

start_time_utc: 2026-05-08T11:18:41Z
end_time_utc: 2026-05-08T13:04:57Z
stop_reason: manual_ctrl_c_power_bank_80_percent_used_20_percent_remaining

total_images: 660564
elapsed_seconds: 6375.730516591
throughput_fps: 103.60601005344763
latency_median_ms: 9.41992
latency_p95_ms: 9.44141
latency_min_ms: 9.27344
latency_max_ms: 24.061

external_logger_archive: /home/agxhasan/Benchmark/battery/for_agx/runs/agx/agx_segformer_b0_fp16_640_20260508T111833Z/external_logger/agx_segformer_b0_fp16_640_20260508T111833Z.tar.gz
external_window_start_utc: 2026-05-08T11:18:41.022007+00:00
external_window_end_utc: 2026-05-08T13:04:56.021564+00:00
external_samples: 63751
external_energy_j: 211676.30427348166
external_energy_wh: 58.79897340930046
external_joules_per_image: 0.32044783590005155
external_avg_power_w: 33.20412846790754
external_mean_power_w: 33.20389029191698
external_peak_power_w: 41.78
external_mean_temp_c: 25.03747944345667
external_max_temp_c: 25.8906

clean_run_checks:
- RUN_COMPLETE event present
- SUMMARY_WRITTEN event present
- INFERENCE_END event present
- EXTERNAL_LOGGER_DOWNLOAD event present
- EXTERNAL_LOGGER_STOP_RESPONSE ok
- stderr.log empty
- no ERROR fatal failed Traceback or Exception in stdout stderr events
- inference_latency.csv rows match total_images plus header
- child loops with FINAL_SUMMARY: 99 of 99
- noise guard enabled and restored wifi bluetooth display and services

primary_files:
- summary_json: /home/agxhasan/Benchmark/battery/for_agx/runs/agx/agx_segformer_b0_fp16_640_20260508T111833Z/summary.json
- summary_md: /home/agxhasan/Benchmark/battery/for_agx/runs/agx/agx_segformer_b0_fp16_640_20260508T111833Z/summary.md
- inference_summary_csv: /home/agxhasan/Benchmark/battery/for_agx/runs/agx/agx_segformer_b0_fp16_640_20260508T111833Z/inference_summary.csv
- inference_latency_csv: /home/agxhasan/Benchmark/battery/for_agx/runs/agx/agx_segformer_b0_fp16_640_20260508T111833Z/inference_latency.csv
- events_csv: /home/agxhasan/Benchmark/battery/for_agx/runs/agx/agx_segformer_b0_fp16_640_20260508T111833Z/events.csv
- external_logger_archive: /home/agxhasan/Benchmark/battery/for_agx/runs/agx/agx_segformer_b0_fp16_640_20260508T111833Z/external_logger/agx_segformer_b0_fp16_640_20260508T111833Z.tar.gz
- noise_guard_summary: /home/agxhasan/Benchmark/battery/for_agx/runs/agx/agx_segformer_b0_fp16_640_20260508T111833Z/noise_guard_summary.json

file_hashes:
- summary_json_sha256: 921a2fea48e002c6a439072f63128d027031af8e3f39557b1af4d18b3362ff4a
- inference_summary_csv_sha256: ba2cbdfd14f4f786cc4bbdb7254c0f852456ac7c99ef2cecc18428c283382281
- external_logger_archive_sha256: e54bae0f6c99fdf3f6f43a03c0211991afa8317c6afefd3d462e295709cd7a97
