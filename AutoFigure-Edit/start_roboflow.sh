#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ACTIVATE="$ROOT_DIR/figure/bin/activate"
CRS_ENV_FILE="$SCRIPT_DIR/.env.crs"

load_export_from_bashrc() {
  local var_name="$1"
  local bashrc="${HOME}/.bashrc"
  local line value

  if [ -n "${!var_name:-}" ]; then
    return
  fi
  if [ ! -f "$bashrc" ]; then
    return
  fi

  line="$(grep -E "^export ${var_name}=" "$bashrc" | tail -n 1 || true)"
  if [ -z "$line" ]; then
    return
  fi

  value="${line#*=}"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  if [ -n "$value" ]; then
    export "${var_name}=${value}"
  fi
}

if [ -f /etc/network_turbo ]; then
  # Optional network acceleration for GitHub/Hugging Face/API endpoints.
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

if [ -f "$CRS_ENV_FILE" ]; then
  # Optional local CRS settings (key/base_url/models).
  # shellcheck disable=SC1090
  source "$CRS_ENV_FILE"
fi

if [ ! -f "$ENV_ACTIVATE" ]; then
  echo "Virtualenv not found: $ENV_ACTIVATE"
  echo "Expected environment path: $ROOT_DIR/figure"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_ACTIVATE"

load_export_from_bashrc "ROBOFLOW_API_KEY"
load_export_from_bashrc "BIANXIE_API_KEY"
load_export_from_bashrc "OPENROUTER_API_KEY"
load_export_from_bashrc "CRS_API_KEY"
load_export_from_bashrc "CRS_BASE_URL"
load_export_from_bashrc "CRS_SVG_MODEL"
: "${ROBOFLOW_API_KEY:=Vxe4NqybbwczubYJyMP4}"
export ROBOFLOW_API_KEY
: "${CRS_API_KEY:=sk-zRCw7PibMrd26IU47W0bogK5TQezqu6KfVw5fwGMyivygXyT}"
export CRS_API_KEY
export CRS_BASE_URL="${CRS_BASE_URL:-http://bruder.yukinoapi.com/v1}"
export CRS_SVG_MODEL="${CRS_SVG_MODEL:-[稳定1]gemini-3-pro-preview}"

if [ -z "${ROBOFLOW_API_KEY:-}" ]; then
  echo "ROBOFLOW_API_KEY is not set."
  echo "Set it first, e.g.:"
  echo "  export ROBOFLOW_API_KEY='your_key_here'"
  exit 1
fi

cd "$SCRIPT_DIR"
exec python server.py
