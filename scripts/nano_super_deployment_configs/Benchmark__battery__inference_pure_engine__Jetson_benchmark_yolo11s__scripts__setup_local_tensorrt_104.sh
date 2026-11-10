#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common_env.sh"

TRT_VERSION="${TRT_VERSION:-${PRIVATE_LAN_IP}-1+cuda12.6}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$BENCHMARK_ROOT/.downloads/tensorrt104}"
INSTALL_ROOT="${INSTALL_ROOT:-$LOCAL_TENSORRT_ROOT}"
CLEAN_DOWNLOADS="${CLEAN_DOWNLOADS:-1}"

RUNTIME_DEBS=(
  "libnvinfer-bin_${TRT_VERSION}_arm64.deb"
  "libnvinfer10_${TRT_VERSION}_arm64.deb"
  "libnvinfer-plugin10_${TRT_VERSION}_arm64.deb"
  "libnvonnxparsers10_${TRT_VERSION}_arm64.deb"
  "libnvinfer-lean10_${TRT_VERSION}_arm64.deb"
  "libnvinfer-vc-plugin10_${TRT_VERSION}_arm64.deb"
  "libnvinfer-dispatch10_${TRT_VERSION}_arm64.deb"
)

HEADER_DEBS=(
  "libnvinfer-headers-dev_${TRT_VERSION}_arm64.deb"
  "libnvinfer-headers-plugin-dev_${TRT_VERSION}_arm64.deb"
)

ALL_PACKAGES=(
  "libnvinfer-bin=${TRT_VERSION}"
  "libnvinfer10=${TRT_VERSION}"
  "libnvinfer-plugin10=${TRT_VERSION}"
  "libnvonnxparsers10=${TRT_VERSION}"
  "libnvinfer-lean10=${TRT_VERSION}"
  "libnvinfer-vc-plugin10=${TRT_VERSION}"
  "libnvinfer-dispatch10=${TRT_VERSION}"
  "libnvinfer-headers-dev=${TRT_VERSION}"
  "libnvinfer-headers-plugin-dev=${TRT_VERSION}"
)

if [[ -x "$INSTALL_ROOT/usr/src/tensorrt/bin/trtexec" && -d "$INSTALL_ROOT/usr/lib/aarch64-linux-gnu" ]]; then
  echo "[INFO] Local TensorRT root already ready: $INSTALL_ROOT"
  exit 0
fi

mkdir -p "$DOWNLOAD_DIR" "$INSTALL_ROOT"

echo "[INFO] Downloading TensorRT 10.4 runtime subset into $DOWNLOAD_DIR"
(cd "$DOWNLOAD_DIR" && apt-get download "${ALL_PACKAGES[@]}")

echo "[INFO] Extracting runtime subset into $INSTALL_ROOT"
for deb in "${RUNTIME_DEBS[@]}" "${HEADER_DEBS[@]}"; do
  dpkg-deb -x "$DOWNLOAD_DIR/$deb" "$INSTALL_ROOT"
done

if [[ "$CLEAN_DOWNLOADS" == "1" ]]; then
  rm -f "$DOWNLOAD_DIR"/*.deb
fi

if [[ ! -x "$INSTALL_ROOT/usr/src/tensorrt/bin/trtexec" ]]; then
  echo "Local trtexec was not extracted correctly under $INSTALL_ROOT" >&2
  exit 1
fi

cat <<EOF
[OK] Local TensorRT 10.4 runtime prepared
  root: $INSTALL_ROOT
  trtexec: $INSTALL_ROOT/usr/src/tensorrt/bin/trtexec
  libs: $INSTALL_ROOT/usr/lib/aarch64-linux-gnu
EOF
