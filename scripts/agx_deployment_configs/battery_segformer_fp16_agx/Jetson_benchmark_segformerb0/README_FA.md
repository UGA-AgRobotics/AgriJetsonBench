# راهنمای فارسی: نسخه‌ی live dashboard + OLED برای benchmark روی Jetson AGX Orin با Raspberry Pi 5 logger

این README برای layout فعلی workspace به‌روز شده و فرض می‌کند package و bundle مدل داخل همین مسیرها هستند:

```text
~/Benchmark/Jetson_benchmark_segformerb0
~/Benchmark/segformerb0/inference_bundle
~/miniconda3
conda env: SAM
```

پیش‌فرض scriptها حالا این‌ها را خودکار فرض می‌کنند:

- package root: `~/Benchmark/Jetson_benchmark_segformerb0`
- model bundle: `~/Benchmark/segformerb0/inference_bundle`
- output root:
  - benchmark: `~/Benchmark/runs/jetson_benchmark_runs`
  - calibration: `~/Benchmark/runs/jetson_calibration_runs`
  - Pi logger: `~/Benchmark/runs/jetson_external_logs`
- conda env: `SAM`
- Pi venv: `~/envs/jetson_pi_logger`

اگر Miniconda یا bundle در جای دیگری است، موقع اجرای scriptها این متغیرها را override کن:

- `MINICONDA_ROOT`
- `CONDA_ENV_NAME`
- `INFERENCE_BUNDLE_ROOT`
- `MODEL_PATH`
- `ACCURACY_MODEL`
- `DATASET_INPUT`
- `OUT_ROOT`

برای Raspberry Pi logger، scriptها حالا به‌صورت خودکار اگر venv زیر موجود باشد همان را ترجیح می‌دهند:

```text
~/envs/jetson_pi_logger
```

یعنی روی Pi لازم نیست env جدید بسازی؛ `setup_pi_env.sh` فقط packageهای لازم را داخل همان env نصب/به‌روزرسانی می‌کند.

## 0) چه چیزهایی در این package هست؟

- dashboard زنده روی Raspberry Pi logger
- OLED SSD1306 روی Jetson
- quiet mode / noise guard روی Jetson
- benchmark پیشنهادی با warm-up و measured window
- calibration suite
- merge و summarize tooling

quiet mode به‌صورت پیش‌فرض در scriptهای benchmark و calibration روشن است و با `scripts/run_with_sudo_keepalive.sh` اجرا می‌شود تا restore آخر کار جا نماند.

### تست مستقل quiet mode

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
PI_LOGGER_URL=http://${LOGGER_DASHBOARD_HOST}:8000 \
bash scripts/test_noise_guard.sh
```

### اگر quiet mode را نخواهی

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
NOISE_GUARD_ENABLE=0 bash scripts/example_run_agx_native.sh
```

### اگر Wi-Fi باید روشن بماند

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
NOISE_GUARD_DISABLE_WIFI=0 bash scripts/example_run_agx_native.sh
```

### اگر DPMS مانیتور را نمی‌خواهی خاموش کند

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
NOISE_GUARD_DISPLAY_OFF=0 bash scripts/example_run_agx_native.sh
```

## 1) wiring

### Raspberry Pi 5: INA260 و TMP117 / TMP119

- `Pin 1` = 3.3V
- `Pin 3` = SDA
- `Pin 5` = SCL
- `Pin 6` = GND

INA260:

- `VCC -> Pin 1`
- `GND -> Pin 6`
- `SDA -> Pin 3`
- `SCL -> Pin 5`

TMP117 یا TMP119:

- `3.3V -> Pin 1`
- `GND -> Pin 6`
- `SDA -> Pin 3`
- `SCL -> Pin 5`

مسیر برق Jetson از داخل INA260:

```text
Adapter +   -> INA260 VIN+
INA260 VIN- -> Jetson DC +
Adapter -   -> Jetson DC -
```

### Jetson: OLED SSD1306 I2C 128x32

- `GND -> Jetson pin 6`
- `VCC -> Jetson pin 1`
- `SCL -> Jetson pin 5`
- `SDA -> Jetson pin 3`

bus معمولاً `7` است، ولی package auto-detect هم دارد.

## 2) راه‌اندازی Raspberry Pi 5 logger

### 2.1 نصب env

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
bash scripts/setup_pi_env.sh
```

اگر `~/envs/jetson_pi_logger` وجود داشته باشد، این script همان venv را استفاده می‌کند. اگر آن venv نباشد، روی fallback می‌رود سراغ conda env `SAM`.

### 2.2 فعال بودن I2C

اگر لازم بود:

```bash
sudo nano /boot/firmware/config.txt
```

و این خط را داشته باش:

```text
dtparam=i2c_arm=on
```

بعد:

```bash
sudo reboot
```

### 2.3 تست I2C و سنسورها

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
bash scripts/check_pi_i2c.sh
source ~/envs/jetson_pi_logger/bin/activate
python pi_logger/scan_i2c_all.py
```

اگر venv تو path دیگری دارد، `~/envs/jetson_pi_logger` را با path واقعی‌ات عوض کن.

### 2.4 اجرای logger + dashboard

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
I2C_BUS=auto bash scripts/run_pi_logger.sh
```

### 2.5 تست health و live API

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
bash scripts/check_logger_health.sh http://127.0.0.1:8000
```

### 2.6 باز کردن dashboard

روی Pi یا لپ‌تاپ یا گوشی:

```text
http://PI-IP:8000/
```

مثال:

```text
http://${LOGGER_DASHBOARD_HOST}:8000/
```

dashboard rooye mobile ham ghabele estefade ast va filehaye PWA/web app dar `pi_logger/web/` gharar دارند.

### 2.7 اگر می‌خواهی logger به‌صورت service بالا بیاید

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
sudo cp service/jetson_external_logger@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now jetson_external_logger@$USER
sudo systemctl status jetson_external_logger@$USER
```

service فعلی مسیر package را `~/Benchmark/...` فرض می‌کند و defaultan هم اول سعی می‌کند از `~/envs/jetson_pi_logger` استفاده کند.

### 2.8 export kardan-e subset-e lazem baraye Pi5

agar bekhahi faghat filehaye lazem ro too yek folder joda baraye enteghal be Pi5 dashte bashi:

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
bash scripts/prepare_pi5_package.sh ~/Benchmark/Pi5_package
```

bad az transfer, behtarin halat in ast ke folder ra rooye Pi dar `~/Pi5_package` bogzari.

## 3) راه‌اندازی Jetson AGX Orin

### 3.1 نصب env

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
bash scripts/setup_jetson_env.sh
```

نکته:

- این script عمداً `torch` را auto-install نمی‌کند.
- باید stack سازگار با Jetson را از قبل داشته باشی.

### 3.2 bundle مدل و دیتاست

اگر همین workspace فعلی را داری، benchmark scriptها به‌صورت پیش‌فرض این مسیر را پیدا می‌کنند:

```text
~/Benchmark/segformerb0/inference_bundle
```

پس اگر model و metadata و zip دیتاست همین‌جا هستند، لازم نیست pathها را دستی بدهی.

### 3.3 اگر بخواهی یک bundle آماده در مسیر دیگری بسازی

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
bash scripts/prepare_inference_bundle.sh ~/Benchmark/segformerb0 ~/Benchmark/prepared_bundle
```

خروجی در `~/Benchmark/prepared_bundle` ساخته می‌شود و شامل model و metadata و dataset آماده است.

### 3.4 ساخت TensorRT engine

اگر بخواهی هر 3 precision را با یک command بسازی:

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
bash scripts/build_engines_int8_fp16_fp32.sh
```

رفتار این script:

- `fp16` را می‌سازد اگر `best_fp16.engine` موجود نباشد
- `fp32` را می‌سازد اگر `best_fp32.engine` موجود نباشد
- `int8` را هم با calibration cache می‌سازد اگر artifactهای آن موجود نباشند
- اگر artifactها از قبل موجود باشند، آن‌ها را skip می‌کند

اگر بخواهی همه‌چیز را از نو rebuild کند:

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
FORCE_REBUILD=1 bash scripts/build_engines_int8_fp16_fp32.sh
```

بعد از ساخت engineها، scriptهای benchmark به‌صورت خودکار `best_fp16.engine`، `best_fp32.engine` و در صورت فعال بودن `best_int8.engine` را به‌عنوان measured model استفاده می‌کنند. برای accuracy pass هنوز `best.pt` یا در fallback `best.onnx` استفاده می‌شود.

## 4) تست OLED

### 4.1 تست سریع با script

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
OLED_BUS=7 OLED_ADDR=0x3C bash scripts/test_jetson_oled.sh
```

### 4.2 اجرای مستقیم tester

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SAM
python tools/jetson_oled_dl_tester.py --bus 7 --addr 0x3C --scenario ok
```

برای auto-detect:

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SAM
python tools/jetson_oled_dl_tester.py --scenario menu
```

## 5) اجرای benchmark

### 5.0 command اصلی benchmark برای report کامل

الان command اصلی این package این script است:

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
INT8_ENABLE=1 \
NOISE_GUARD_ENABLE=1 \
OLED_BUS=7 \
OLED_ADDR=0x3C \
PI_LOGGER_URL=http://${LOGGER_DASHBOARD_HOST}:8000 \
bash scripts/run_current_mode_int8_fp16_fp32.sh
```

رفتار پیش‌فرض این command:

- power profile را عوض نمی‌کند و فقط mode فعلی Jetson را detect می‌کند
- اگر engineهای لازم موجود نباشند، آن‌ها را از `best.onnx` به‌صورت خودکار build می‌کند
- برای همان power profile فعلی، همه precisionهای موجود را run می‌کند
  - `fp16`
  - `fp32`
  - `int8` اگر `INT8_ENABLE=1` داده باشی، در صورت نیاز auto-build هم می‌شود
- برای هر precision، `3 repeat` report-grade می‌گیرد وقتی `REPORT_MODE=1` باشد
- defaultan **دو variant جدا** می‌سازد:
  - `standard`
  - `per_image`

تفاوت این دو variant:

- `standard`
  - همان summaryهای کلی report را نگه می‌دارد
  - هیچ artifact per-image داخلش نگه نمی‌دارد
- `per_image`
  - علاوه بر summaryهای کلی، artifactهای per-image را هم نگه می‌دارد
  - مثل `per_image_latency.csv` و `per_image_metrics.csv`

اگر بخواهی فقط یکی از variantها را اجرا کنی:

```bash
RUN_VARIANTS=standard ...
```

یا:

```bash
RUN_VARIANTS=per_image ...
```

### 5.1 native benchmark با dashboard و OLED

اگر bundle در مسیر پیش‌فرض `~/Benchmark/segformerb0/inference_bundle` است:

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
PI_LOGGER_URL=http://${LOGGER_DASHBOARD_HOST}:8000 \
OUT_ROOT=~/Benchmark/runs/jetson_benchmark_runs \
OLED_ENABLE=1 \
OLED_BUS=7 \
OLED_ADDR=0x3C \
bash scripts/example_run_agx_native.sh
```

اگر engine نساخته‌ای، script به‌صورت خودکار روی `best.onnx` یا `best.pt` fallback می‌کند.

### 5.2 اگر بخواهی مسیرها را explicit بدهی

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
MODEL_PATH=~/Benchmark/segformerb0/inference_bundle/model/best_fp16.engine \
ACCURACY_MODEL=~/Benchmark/segformerb0/inference_bundle/model/best.pt \
DATASET_INPUT=~/Benchmark/segformerb0/inference_bundle \
PI_LOGGER_URL=http://${LOGGER_DASHBOARD_HOST}:8000 \
OUT_ROOT=~/Benchmark/runs/jetson_benchmark_runs \
OLED_ENABLE=1 \
OLED_BUS=7 \
OLED_ADDR=0x3C \
bash scripts/example_run_agx_native.sh
```

`DATASET_INPUT` می‌تواند یکی از این‌ها باشد:

- root خود `inference_bundle`
- dataset zip
- dataset extract شده

### 5.3 matched-budget benchmark

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
MATCHED_NVP_MODE=<mode_id> \
PI_LOGGER_URL=http://${LOGGER_DASHBOARD_HOST}:8000 \
OUT_ROOT=~/Benchmark/runs/jetson_benchmark_runs \
OLED_ENABLE=1 \
OLED_BUS=7 \
OLED_ADDR=0x3C \
bash scripts/example_run_agx_matched15w.sh
```

### 5.4 sustained 30-minute run

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
PI_LOGGER_URL=http://${LOGGER_DASHBOARD_HOST}:8000 \
OUT_ROOT=~/Benchmark/runs/jetson_benchmark_runs \
RUN_LABEL=agx_orin_segformerb0_fp16_topconfig_30min \
OLED_ENABLE=1 \
OLED_BUS=7 \
OLED_ADDR=0x3C \
bash scripts/example_sustained_top_config.sh
```

## 6) calibration suite

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
PI_LOGGER_URL=http://${LOGGER_DASHBOARD_HOST}:8000 \
OUT_ROOT=~/Benchmark/runs/jetson_calibration_runs \
bash scripts/example_calibration_agx.sh
```

برای calibration سناریوهای `gpu` و `mixed`، model و dataset به‌صورت خودکار از bundle محلی resolve می‌شوند.

## 7) smoke test مستقیم با Python

اگر بخواهی مستقیم `run_benchmark.py` را صدا بزنی:

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate SAM
python jetson/run_benchmark.py \
  --model ~/Benchmark/segformerb0/inference_bundle/model/best.onnx \
  --dataset-root ~/Benchmark/segformerb0/inference_bundle \
  --run-name smoke_dashboard_oled \
  --board-name "Jetson AGX Orin 64GB" \
  --output-root ~/Benchmark/runs/jetson_benchmark_runs \
  --device 0 \
  --repeats 1 \
  --warmup-images 4 \
  --warmup-seconds 10 \
  --min-images 32 \
  --min-duration-s 20 \
  --pi-logger-url http://${LOGGER_DASHBOARD_HOST}:8000 \
  --expect-external-logger \
  --external-sample-hz 10 \
  --tegrastats-interval-ms 200 \
  --internal-interval-ms 200 \
  --start-delay-s 3 \
  --power-mode-label smoke \
  --noise-guard \
  --noise-guard-profile safe \
  --oled-enable \
  --oled-bus 7 \
  --oled-addr 0x3C
```

در حالت direct Python دیگر لازم نیست `--data-yaml` و `--manifest` را بدهی؛ code آن‌ها را از bundle portable پیدا می‌کند.

## 8) dashboard چه چیزهایی نشان می‌دهد؟

- stage
- repeat
- elapsed time
- ETA
- processed images
- external voltage/current/power/temperature
- internal power/thermal/tegrastats
- alertهای `voltage_drop` و `load_drop`

## 9) OLED چه چیزهایی نشان می‌دهد؟

- stage
- repeat
- image progress
- ETA
- latency آخرین image
- finished
- error

## 10) فایل‌های مهم خروجی

benchmark:

- `FINAL_SUMMARY.json`
- `aggregate_repeat_summary.csv`
- `benchmark_runs/repeat_*/repeat_summary.json`
- `benchmark_runs/repeat_*/external/`
- `benchmark_runs/repeat_*/internal/`

در command اصلی `run_current_mode_int8_fp16_fp32.sh` خروجی‌ها داخل دو variant جدا قرار می‌گیرند:

- `standard/`
- `per_image/`

در variant `standard` این‌ها intentionally وجود ندارند:

- `benchmark_runs/repeat_*/per_image_latency.csv`
- `benchmark_runs/repeat_*/per_image_metrics.csv`
- `benchmark_runs/repeat_*/trtexec_times.json`

در variant `per_image` این‌ها وجود دارند و مخصوص تحلیل per-image هستند.

calibration:

- `calibration_summary.json`
- external archive logs
- internal telemetry logs

## 11) جمع‌بندی عملی

برای layout فعلی workspace، commandهای اصلی که معمولاً کافی هستند این‌ها هستند:

روی Pi:

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
bash scripts/setup_pi_env.sh
I2C_BUS=auto bash scripts/run_pi_logger.sh
```

روی Jetson:

```bash
cd ~/Benchmark/Jetson_benchmark_segformerb0
bash scripts/setup_jetson_env.sh
INT8_ENABLE=1 PI_LOGGER_URL=http://${LOGGER_DASHBOARD_HOST}:8000 OLED_BUS=7 OLED_ADDR=0x3C bash scripts/run_current_mode_int8_fp16_fp32.sh
```

اگر `best_int8.engine` نداشته باشی، همین command فقط `fp16` و `fp32` را اجرا می‌کند. اگر داشته باشی، `int8` را هم به همان session اضافه می‌کند.
