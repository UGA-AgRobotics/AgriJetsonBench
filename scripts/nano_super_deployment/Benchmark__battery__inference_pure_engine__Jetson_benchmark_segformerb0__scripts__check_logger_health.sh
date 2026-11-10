#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${1:-http://127.0.0.1:8000}"
echo "Dashboard: $BASE_URL/"
echo "--- /health ---"
curl -fsSL "$BASE_URL/health" ; echo
echo "--- /scan ---"
curl -fsSL "$BASE_URL/scan" ; echo
echo "--- /read_once ---"
curl -fsSL "$BASE_URL/read_once" ; echo
echo "--- /api/live ---"
curl -fsSL "$BASE_URL/api/live" ; echo
