# Install into the active Python environment. Extras are optional.
[positional-arguments]
install extras='' features='cuda':
    #!/usr/bin/env bash
    set -euo pipefail
    wheels="$(mktemp -d)"
    trap 'rm -rf "$wheels"' EXIT
    export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PWD/apxinf/target/wheel}"
    python -m pip install 'maturin>=1.5,<2'
    extras="$1"
    features="$2"
    set --
    if [[ -n "$features" ]]; then set -- --features "$features"; fi
    python -m maturin build --locked --release --auditwheel skip \
        -m apxinf/crates/apxinf-py/Cargo.toml --interpreter "$(command -v python)" \
        --out "$wheels" "$@"
    python -m pip install --force-reinstall --no-deps "$wheels"/apxinf_py-*.whl
    python -m pip install -e ./apxinf/python/apxinf --config-settings editable_mode=strict
    package='.'
    if [[ -n "$extras" ]]; then package=".[$extras]"; fi
    python -m pip install -e "$package"
    python -m pip check
