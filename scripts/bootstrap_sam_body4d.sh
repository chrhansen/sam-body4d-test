#!/usr/bin/env bash
set -euo pipefail

SAM_BODY4D_REPO="${SAM_BODY4D_REPO:-https://github.com/gaomingqi/sam-body4d.git}"
SAM_BODY4D_REF="${SAM_BODY4D_REF:-master}"
SAM_BODY4D_WORKDIR="${SAM_BODY4D_WORKDIR:-/workspace/sam-body4d}"
SAM_BODY4D_CKPT_ROOT="${SAM_BODY4D_CKPT_ROOT:-/workspace/checkpoints}"
SAM_BODY4D_BATCH_SIZE="${SAM_BODY4D_BATCH_SIZE:-16}"
SAM_BODY4D_ENABLE_COMPLETION="${SAM_BODY4D_ENABLE_COMPLETION:-false}"
GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGINGFACE_API_KEY:-}" ]; then
  export HF_TOKEN="$HUGGINGFACE_API_KEY"
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN or HUGGINGFACE_API_KEY must be set" >&2
  exit 1
fi

install_system_deps() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return 0
  fi

  if [ "$(id -u)" -eq 0 ]; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git ffmpeg libgl1 libglib2.0-0 libegl1 libglvnd0
  elif command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git ffmpeg libgl1 libglib2.0-0 libegl1 libglvnd0
  fi
}

clone_or_update_repo() {
  mkdir -p "$(dirname "$SAM_BODY4D_WORKDIR")"
  if [ ! -d "$SAM_BODY4D_WORKDIR/.git" ]; then
    git clone "$SAM_BODY4D_REPO" "$SAM_BODY4D_WORKDIR"
  fi

  cd "$SAM_BODY4D_WORKDIR"
  git fetch --all --tags

  if git rev-parse --verify --quiet "$SAM_BODY4D_REF" >/dev/null; then
    git checkout "$SAM_BODY4D_REF"
  else
    git checkout -B "$SAM_BODY4D_REF" "origin/$SAM_BODY4D_REF"
  fi

  git pull --ff-only origin "$SAM_BODY4D_REF"
}

patch_upstream_sources() {
  python - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["SAM_BODY4D_WORKDIR"])
utils_path = root / "models" / "sam_3d_body" / "notebook" / "utils.py"
app_path = root / "app.py"

utils_txt = utils_path.read_text(encoding="utf-8")

old_occ = """                # Get bounding box from mask contours
                x, y, w, h = cv2.boundingRect(coords)
                bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)
                # print(f\"Computed bbox from mask: {bbox[0]}\")
                _occ_bbox_list.append(bbox)
"""
new_occ = """                # Get bounding box from mask contours (safe for empty masks)
                if coords is None:
                    bbox = np.array([[0, 0, 1, 1]], dtype=np.float32)
                else:
                    x, y, w, h = cv2.boundingRect(coords)
                    bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)
                # print(f\"Computed bbox from mask: {bbox[0]}\")
                _occ_bbox_list.append(bbox)
"""

old_no_occ = """                # Get bounding box from mask contours
                x, y, w, h = cv2.boundingRect(coords)
                bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)

                # print(f\"Computed bbox from mask: {bbox[0]}\")
                no_occ_bbox_list.append(bbox)
"""
new_no_occ = """                # Get bounding box from mask contours (safe for empty masks)
                if coords is None:
                    bbox = np.array([[0, 0, 1, 1]], dtype=np.float32)
                else:
                    x, y, w, h = cv2.boundingRect(coords)
                    bbox = np.array([[x, y, x + w, y + h]], dtype=np.float32)

                # print(f\"Computed bbox from mask: {bbox[0]}\")
                no_occ_bbox_list.append(bbox)
"""

if old_occ in utils_txt:
    utils_txt = utils_txt.replace(old_occ, new_occ, 1)
if old_no_occ in utils_txt:
    utils_txt = utils_txt.replace(old_no_occ, new_no_occ, 1)

utils_path.write_text(utils_txt, encoding="utf-8")

app_txt = app_path.read_text(encoding="utf-8")
if "demo.launch(show_error=True)" not in app_txt:
    app_txt = app_txt.replace("demo.launch()", "demo.launch(show_error=True)", 1)
app_path.write_text(app_txt, encoding="utf-8")
PY
}

install_python_deps() {
  cd "$SAM_BODY4D_WORKDIR"

  python --version
  python -m pip install --upgrade pip setuptools wheel

  if ! python - <<'PY'
import sys
try:
    import torch
except Exception:
    sys.exit(1)
print(torch.__version__)
if not torch.__version__.startswith("2.7.1"):
    sys.exit(1)
PY
  then
    python -m pip install --no-cache-dir \
      torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
      --index-url https://download.pytorch.org/whl/cu118
  fi

  python -m pip install --no-cache-dir -e models/sam3

  # Install app runtime deps directly to avoid python-version gates on project metadata.
  python -m pip install --no-cache-dir "numpy==1.26.0"
  python -m pip install --no-cache-dir \
    "gradio~=6.0.0" \
    "opencv-python==4.11.0.86" \
    "einops~=0.8.1" \
    "decord~=0.6.0" \
    "pycocotools~=2.0.10" \
    "psutil~=7.1.3" \
    "braceexpand~=0.1.7" \
    "roma~=1.5.4" \
    "omegaconf~=2.3.0" \
    "pytorch_lightning" \
    "yacs~=0.1.8" \
    "matplotlib~=3.10.7" \
    "cloudpickle~=3.1.2" \
    "fvcore~=0.1.5.post20221221" \
    "pyrender~=0.1.45" \
    "termcolor~=3.2.0" \
    "diffusers==0.29.1" \
    "transformers~=4.57.3" \
    "accelerate~=1.12.0" \
    "imageio[ffmpeg]" \
    "scipy<1.17" \
    "MoGe @ git+https://github.com/microsoft/MoGe.git"
}

setup_checkpoints() {
  cd "$SAM_BODY4D_WORKDIR"
  python scripts/setup.py --ckpt-root "$SAM_BODY4D_CKPT_ROOT"

  local enable_val
  enable_val="false"
  case "${SAM_BODY4D_ENABLE_COMPLETION,,}" in
    true|1|yes|y|on) enable_val="true" ;;
  esac

  sed -i "s/^  batch_size: .*/  batch_size: ${SAM_BODY4D_BATCH_SIZE}/" configs/body4d.yaml
  sed -i "s/^  enable: .*/  enable: ${enable_val}/" configs/body4d.yaml
}

launch_app() {
  cd "$SAM_BODY4D_WORKDIR"
  export GRADIO_SERVER_NAME
  export GRADIO_SERVER_PORT
  exec python -u app.py
}

main() {
  install_system_deps
  clone_or_update_repo
  patch_upstream_sources
  install_python_deps
  setup_checkpoints
  launch_app
}

main "$@"
