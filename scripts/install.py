"""Build and install the pinned engine and robot package in this Python environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="build for packaging and adapter tests, without CUDA inference",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        choices=["serve", "libero", "drive", "robosuite", "isaaclab", "lerobot", "dev"],
        help="optional dependencies; repeat to install multiple extras",
    )
    args = parser.parse_args()
    engine = ROOT / "apxinf"
    if not (engine / "crates/apxinf-py/Cargo.toml").exists():
        run("git", "submodule", "update", "--init", "--recursive")
    run(sys.executable, "-m", "pip", "install", "maturin>=1.5,<2.0")
    with tempfile.TemporaryDirectory(prefix="apxinf-wheels-") as directory:
        command = [
            sys.executable,
            "-m",
            "maturin",
            "build",
            "--locked",
            "--release",
            "--auditwheel",
            "skip",
            "--manifest-path",
            str(engine / "crates/apxinf-py/Cargo.toml"),
            "--interpreter",
            sys.executable,
            "--out",
            directory,
        ]
        if not args.cpu_only:
            command.extend(["--features", "cuda"])
        run(*command)
        wheels = list(Path(directory).glob("apxinf-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(
                "Expected one combined apxinf wheel; update the engine submodule"
            )
        # Old split distributions own the same import path as the combined wheel.
        try:
            importlib.metadata.distribution("apxinf-py")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            run(sys.executable, "-m", "pip", "uninstall", "-y", "apxinf-py")
        package = str(ROOT)
        if args.extra:
            package += "[" + ",".join(args.extra) + "]"
        run(
            sys.executable, "-m", "pip", "install", "--upgrade", str(wheels[0]), package
        )
    run(sys.executable, "-m", "pip", "check")


if __name__ == "__main__":
    main()
