#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

TARGET_DIR="${1:-$BENCHMARK_ROOT/Pi5_package}"
TARGET_DIR="$(python3 - <<'PY' "$TARGET_DIR"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

mkdir -p \
  "$TARGET_DIR" \
  "$TARGET_DIR/pi_logger" \
  "$TARGET_DIR/jetson" \
  "$TARGET_DIR/scripts" \
  "$TARGET_DIR/service" \
  "$TARGET_DIR/chrony"

cp -a "$PKG_ROOT/pi_logger/." "$TARGET_DIR/pi_logger/"
cp -a "$PKG_ROOT/jetson/__init__.py" "$TARGET_DIR/jetson/__init__.py"
cp -a "$PKG_ROOT/jetson/local_paths.py" "$TARGET_DIR/jetson/local_paths.py"
cp -a "$PKG_ROOT/requirements_pi.txt" "$TARGET_DIR/requirements_pi.txt"
cp -a "$PKG_ROOT/chrony/chrony_server_pi.conf.example" "$TARGET_DIR/chrony/chrony_server_pi.conf.example"

for script_name in \
  common_env.sh \
  setup_pi_env.sh \
  run_pi_logger.sh \
  check_pi_i2c.sh \
  check_logger_health.sh
do
  cp -a "$PKG_ROOT/scripts/${script_name}" "$TARGET_DIR/scripts/${script_name}"
done

cp -a "$PKG_ROOT/service/pi5_external_logger@.service" "$TARGET_DIR/service/pi5_external_logger@.service"

cat > "$TARGET_DIR/README_FA.md" <<'EOF'
# Pi5_package

In folder faghat chizhayi gharar gerefte ke rooye Raspberry Pi 5 baraye logger/dashboard lazem dari.

## Koja copy konam?

Behtarin halat in ast ke hamin folder ra rooye Pi dar in path bogzari:

```text
~/Pi5_package
```

Age path ra avaz kardi, file service ra ham bayad motenaseb koni.

## In package chie?

- logger baraye INA260 + TMP117/TMP119
- dashboard web rooye port `8000`
- PWA-like mobile view
- config mesle `chrony_server_pi.conf.example`

## 1) env ro amade kon

```bash
cd ~/Pi5_package
bash scripts/setup_pi_env.sh
```

In script defaultan agar `~/envs/jetson_pi_logger` vojood dashte bashad, haman ra estefade mikonad va packagehaye lazem ra toosh update/install mikonad. Pas niyazi be sakhte env jadid nist.

## 2) I2C ro check kon

```bash
cd ~/Pi5_package
bash scripts/check_pi_i2c.sh
source ~/envs/jetson_pi_logger/bin/activate
python pi_logger/scan_i2c_all.py
```

## 3) logger ro run kon

```bash
cd ~/Pi5_package
I2C_BUS=auto bash scripts/run_pi_logger.sh
```

## 4) health check

```bash
cd ~/Pi5_package
bash scripts/check_logger_health.sh http://127.0.0.1:8000
```

## 5) dashboard ro baz kon

Az rooye mobile ya laptop:

```text
http://PI-IP:8000/
```

## 6) optional: systemd service

```bash
sudo cp service/pi5_external_logger@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pi5_external_logger@$USER
sudo systemctl status pi5_external_logger@$USER
```
EOF

cat > "$TARGET_DIR/README.md" <<'EOF'
# Pi5_package

This folder contains the Raspberry Pi 5 side only:

- external INA260/TMP117 logger
- web dashboard on port `8000`
- installable mobile/PWA-like view

Recommended target path on the Pi:

```text
~/Pi5_package
```

Quick start:

```bash
cd ~/Pi5_package
bash scripts/setup_pi_env.sh
I2C_BUS=auto bash scripts/run_pi_logger.sh
```

If `~/envs/jetson_pi_logger` already exists on the Pi, the scripts will use that venv automatically instead of creating a new environment.
EOF

cat > "$TARGET_DIR/package_manifest.json" <<EOF
{
  "generated_from": "$PKG_ROOT",
  "target_dir": "$TARGET_DIR",
  "timestamp_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "contents": [
    "README.md",
    "README_FA.md",
    "requirements_pi.txt",
    "chrony/chrony_server_pi.conf.example",
    "jetson/__init__.py",
    "jetson/local_paths.py",
    "pi_logger/**",
    "scripts/common_env.sh",
    "scripts/setup_pi_env.sh",
    "scripts/run_pi_logger.sh",
    "scripts/check_pi_i2c.sh",
    "scripts/check_logger_health.sh",
    "service/pi5_external_logger@.service"
  ]
}
EOF

chmod +x \
  "$TARGET_DIR/scripts/setup_pi_env.sh" \
  "$TARGET_DIR/scripts/run_pi_logger.sh" \
  "$TARGET_DIR/scripts/check_pi_i2c.sh" \
  "$TARGET_DIR/scripts/check_logger_health.sh"

find "$TARGET_DIR" -depth -type d -name '__pycache__' -exec rm -rf {} +
find "$TARGET_DIR" -type f -name '*.pyc' -delete

echo "[OK] Pi5 package آماده شد:"
echo "  $TARGET_DIR"
