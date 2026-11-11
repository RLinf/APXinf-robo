#!/usr/bin/env bash
# Deploy Pi05 LIBERO on a D-Robotics S600: apx-drobotics, and apxinf-robo above it.
#
# Run on the board from a checkout of this repository. ApxInf is not needed there,
# and neither is the apxinf/ submodule:
#
#   bash apx-drobotics/scripts/deploy.sh [--root /opt/d-robotics-llm/2.0.0] [--with-libero]
#
# Steps, each skipped when its result is already in place:
#   1. Download the D-Robotics LLM SDK v2.0.0, check its SHA-256 against the
#      contract, and unpack the board runtime (oellm_runtime_s600) only.
#   2. Download the official Pi05_libero model, check it, and unpack it into
#      the SDK's model directory.
#   3. Create a Python virtual environment with the pinned packages, and install
#      apx-drobotics and apxinf-robo into it.
#   4. Build the C++ worker (apx-drobotics/worker) against the SDK.
#   5. Check the deployment against the contract.
#   6. With --with-libero: install the simulator so `eval-libero --backend
#      in-process` runs on the board, and write the wrapper that starts it.
# A failed check stops the script before anything is reported as ready.
#
# --with-libero is off by default because serving observations does not need a
# simulator, and the simulator costs about 5 GB of disk. It needs packages from
# apt (see LIBERO_APT_PACKAGES) and roughly 1.5 GB of downloads.
set -euo pipefail

PACKAGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$PACKAGE/.." && pwd)"
ROOT=/opt/d-robotics-llm/2.0.0
WITH_LIBERO=0
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --with-libero) WITH_LIBERO=1; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

SDK_URL=https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/D-Robotics_LLM_2.0.0_S600_S100_SDK.tar.gz
MODEL_URL=https://d-robotics-aitoolchain.oss-cn-beijing.aliyuncs.com/llm/2.0.0/s600_models/Pi05_libero.zip
SDK_ARCHIVE_DIR=D-Robotics_LLM_2.0.0_S600_S100_SDK
SDK_MEMBER="$SDK_ARCHIVE_DIR/oellm_runtime_s600"
SDK_DIR="$ROOT/$SDK_MEMBER"
CONTRACT="$PACKAGE/src/apx_drobotics/contracts/pi05_libero_w8.json"
VENV="$ROOT/venv-apx-drobotics"
BUILD="$PACKAGE/build/worker"
DOWNLOADS="$ROOT/.downloads"

# In-process LIBERO (--with-libero only). LIBERO is not on PyPI, so it is cloned
# at a pinned commit and left uninstalled as a package: a .pth puts the checkout
# on sys.path, which is how the PC environment that produced the baselines runs
# it as well. The mirror is used because github.com itself is not reliably
# reachable from the board.
LIBERO_REPO=https://ghfast.top/https://github.com/Lifelong-Robot-Learning/LIBERO.git
LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
LIBERO_DIR="$ROOT/LIBERO"
LIBERO_CONFIG_DIR="$ROOT/libero-config"   # LIBERO wants a directory holding config.yaml
EVAL_SCRIPT="$ROOT/run-libero-in-process.sh"

# System packages the simulator needs. libosmesa6 is what renders on this board.
# Nothing here is compiled: every Python package in requirements-board-libero.txt
# has an aarch64 cp312 wheel, which is why mujoco is pinned to 3.x.
LIBERO_APT_PACKAGES="libosmesa6"

log() { printf '[drobotics-deploy] %s\n' "$*"; }

contract_value() {
  python3 -I -c 'import json,sys; v=json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."): v=v[k]
print(v)' "$CONTRACT" "$1"
}

# download URL DESTINATION SHA256
download() {
  local url="$1" destination="$2" expected="$3" actual
  mkdir -p "$(dirname "$destination")"
  if [ -f "$destination" ] && [ "$(sha256sum "$destination" | cut -d' ' -f1)" = "$expected" ]; then
    log "already downloaded: $destination"
    return
  fi
  log "downloading $url"
  curl -fL --retry 10 --retry-delay 5 --retry-all-errors -C - -o "$destination.part" "$url"
  actual="$(sha256sum "$destination.part" | cut -d' ' -f1)"
  if [ "$actual" != "$expected" ]; then
    echo "SHA-256 mismatch for $url: got $actual, contract says $expected" >&2
    rm -f "$destination.part"
    exit 1
  fi
  mv "$destination.part" "$destination"
}

mkdir -p "$ROOT"

# 1. SDK -----------------------------------------------------------------------
if [ -f "$SDK_DIR/include/oellm_runtime_basic/oellm_runtime.h" ]; then
  log "SDK present: $SDK_DIR"
else
  archive="$DOWNLOADS/$SDK_ARCHIVE_DIR.tar.gz"
  download "$SDK_URL" "$archive" "$(contract_value sdk.archive_sha256)"
  staging="$ROOT/.staging-sdk"
  rm -rf "$staging" && mkdir -p "$staging"
  log "unpacking $SDK_MEMBER"
  # The x86 cross-compiler inside the runtime directory is not used on the board.
  tar -xzf "$archive" -C "$staging" --exclude="$SDK_MEMBER/arm-gnu-toolchain-*" "$SDK_MEMBER"
  chmod -R go-w "$staging"
  mkdir -p "$ROOT/$SDK_ARCHIVE_DIR"
  mv "$staging/$SDK_MEMBER" "$SDK_DIR"
  rm -rf "$staging" "$archive"
fi

# 2. Model ---------------------------------------------------------------------
MODEL_NAME="$(contract_value model.name)"
MODEL_DIR="$SDK_DIR/$(contract_value model.directory)"
if [ -d "$MODEL_DIR" ]; then
  log "model present: $MODEL_DIR"
else
  archive="$DOWNLOADS/$MODEL_NAME.zip"
  download "$MODEL_URL" "$archive" "$(contract_value model.archive_sha256)"
  staging="$(dirname "$MODEL_DIR")/.staging-$MODEL_NAME"
  rm -rf "$staging" && mkdir -p "$staging"
  log "unpacking $MODEL_NAME"
  python3 -I - "$archive" "$staging" "$MODEL_NAME" <<'PY'
import sys, zipfile
from pathlib import PurePosixPath
archive, target, name = sys.argv[1:]
with zipfile.ZipFile(archive) as bundle:
    for member in bundle.namelist():
        parts = PurePosixPath(member).parts
        if not parts or parts[0] != name or ".." in parts or member.startswith("/"):
            sys.exit(f"unexpected path in {archive}: {member}")
    bundle.extractall(target)
PY
  chmod -R go-w "$staging"
  mv "$staging/$MODEL_NAME" "$MODEL_DIR"
  rm -rf "$staging" "$archive"
fi
rmdir "$DOWNLOADS" 2>/dev/null || true

# 3. Python environment --------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
  log "creating $VENV"
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --disable-pip-version-check \
  -r "$PACKAGE/requirements-board.txt"
"$VENV/bin/python" -m pip install --quiet --disable-pip-version-check --no-deps \
  -e "$PACKAGE" -e "$REPO"

# 4. Worker --------------------------------------------------------------------
log "building the worker in $BUILD"
cmake -S "$PACKAGE/worker" -B "$BUILD" -DOELLM_ROOT="$SDK_DIR" >/dev/null
cmake --build "$BUILD" -j"$(nproc)" >/dev/null

# 5. Check ---------------------------------------------------------------------
log "checking the deployment against the contract (hashes about 4 GB of files)"
"$VENV/bin/apx-drobotics" check-deployment --contract "$CONTRACT" --sdk-root "$SDK_DIR"

# 6. In-process LIBERO (opt-in) -------------------------------------------------
if [ "$WITH_LIBERO" = 1 ]; then
  log "installing the in-process simulator"

  missing=""
  for package in $LIBERO_APT_PACKAGES; do
    dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed" || missing="$missing $package"
  done
  if [ -n "$missing" ]; then
    log "installing system packages:$missing"
    apt-get update -qq
    # shellcheck disable=SC2086
    apt-get install -y -qq $missing
  fi

  # requirements-board.txt is passed as a constraint so the simulator install
  # cannot move numpy or pillow, which the letterbox resize depends on.
  # torch is filtered out here and installed second, from the CPU index: PyPI's
  # aarch64 build is a CUDA build and fails at import on a board with no CUDA
  # runtime ("libcudart.so.13: cannot open shared object file"). LIBERO only
  # calls torch.load to read the initial states, so the CPU build is enough.
  TORCH_PIN="$(grep -oE '^torch==[^[:space:]]+' "$PACKAGE/requirements-board-libero.txt")"
  without_torch="$(mktemp)"
  grep -vE '^torch==' "$PACKAGE/requirements-board-libero.txt" > "$without_torch"
  "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check \
    -c "$PACKAGE/requirements-board.txt" -r "$without_torch"
  rm -f "$without_torch"
  "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check \
    --force-reinstall --no-deps \
    --index-url https://download.pytorch.org/whl/cpu "$TORCH_PIN"

  if [ -d "$LIBERO_DIR/.git" ]; then
    log "LIBERO present: $LIBERO_DIR"
  else
    log "cloning LIBERO at $LIBERO_COMMIT"
    rm -rf "$LIBERO_DIR"
    git clone --quiet "$LIBERO_REPO" "$LIBERO_DIR"
    git -C "$LIBERO_DIR" checkout --quiet "$LIBERO_COMMIT"
  fi

  # libero/__init__.py is untracked in the upstream repository, so a fresh clone
  # has no `libero` package and every import fails. The PC checkouts created it
  # by hand; create it here too, idempotently.
  [ -f "$LIBERO_DIR/libero/__init__.py" ] || : > "$LIBERO_DIR/libero/__init__.py"
  # LIBERO reads its asset locations from a config file and prompts on stdin when
  # it is missing, which would hang an unattended run. Point it at the checkout.
  mkdir -p "$LIBERO_CONFIG_DIR"
  cat > "$LIBERO_CONFIG_DIR/config.yaml" <<EOF
benchmark_root: $LIBERO_DIR/libero/libero
bddl_files: $LIBERO_DIR/libero/libero/bddl_files
init_states: $LIBERO_DIR/libero/libero/init_files
datasets: $LIBERO_DIR/datasets
assets: $LIBERO_DIR/libero/libero/assets
EOF
  # The checkout root goes on sys.path rather than being pip-installed: that is
  # how the PC environment runs it, and it keeps LIBERO's own setup.py (whose
  # package discovery misses the nested libero/libero) out of the picture.
  site_packages="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  printf '%s\n' "$LIBERO_DIR" > "$site_packages/libero_checkout.pth"

  # MUJOCO_GL=osmesa, not egl: the board's libEGL is the vendor (hobot) build,
  # which does not export eglQueryDevicesEXT, and mujoco's EGL path needs that
  # extension to enumerate devices. OSMesa renders correctly here.
  cat > "$EVAL_SCRIPT" <<EOF
#!/usr/bin/env bash
# In-process LIBERO evaluation on the S600 board, written by deploy.sh.
# MUJOCO_GL=osmesa: this board's EGL is the vendor build and lacks the
# extension mujoco needs. TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD: the initial states
# are pickled with numpy and recent torch refuses them without it.
set -euo pipefail
export MUJOCO_GL=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_DIR"
exec $VENV/bin/apxinf-robo eval-libero --backend in-process --engine drobotics \\
  --precision w8 --model-dir $SDK_DIR "\$@"
EOF
  chmod +x "$EVAL_SCRIPT"

  # Same MUJOCO_GL as the wrapper: robosuite picks its GL backend at import.
  MUJOCO_GL=osmesa "$VENV/bin/python" - <<'PY'
import sys
for module in ("mujoco", "robosuite", "bddl", "torch", "libero", "libero.libero"):
    __import__(module)
import mujoco, robosuite, torch
print(f"[drobotics-deploy] simulator ready: mujoco {mujoco.__version__}, "
      f"robosuite {robosuite.__version__}, torch {torch.__version__}")
PY
fi

log "ready. Start the service with:"
cat <<EOF
  $VENV/bin/apxinf-robo serve --engine drobotics --robot franka_libero \\
    --model-dir $SDK_DIR --port 8000
EOF
if [ "$WITH_LIBERO" = 1 ]; then
  cat <<EOF
Evaluate in-process on this board with:
  $EVAL_SCRIPT --suite libero_10 --tasks 0 --trials-per-task 1 \\
    --results-jsonl results.jsonl --summary-json summary.json
EOF
fi
