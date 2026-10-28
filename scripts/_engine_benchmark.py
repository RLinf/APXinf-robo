"""Load a benchmark from the pinned engine without duplicating its workload."""
from pathlib import Path
import importlib.util
import os
import sys


_PATH_OPTIONS = frozenset(("--model-dir", "--calibration", "--tactics", "--out", "--binary", "--frames"))


def _absolute_paths(argv, cwd):
    """Keep caller-relative file arguments stable when loading from the engine root."""
    result = list(argv)
    for index, value in enumerate(result[1:], start=1):
        name, separator, inline = value.partition("=")
        if name not in _PATH_OPTIONS:
            continue
        if separator:
            if inline != "-":
                result[index] = f"{name}={Path(cwd, inline).resolve()}"
        elif index + 1 < len(result) and result[index + 1] != "-" and not result[index + 1].startswith("--"):
            result[index + 1] = str(Path(cwd, result[index + 1]).resolve())
    return result


def run(name, *, policy=False):
    root = Path(__file__).resolve().parents[1]
    engine = root / "apxinf"
    path = engine / "scripts" / f"bench_{name}.py"
    if not path.is_file():
        raise RuntimeError(f"missing engine benchmark: {path}; initialize the apxinf submodule")
    previous_cwd = Path.cwd()
    previous_argv = sys.argv
    previous_path = sys.path[:]
    sys.argv = _absolute_paths(sys.argv, previous_cwd)
    try:
        os.chdir(engine)
        sys.path.insert(0, str(root / "src"))
        sys.path.insert(0, str(engine / "scripts"))
        sys.path.insert(0, str(engine / "python/apxinf"))
        spec = importlib.util.spec_from_file_location(f"apxinf_bench_{name}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load engine benchmark: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if policy:
            from apxinf_robo import load_policy
            module.main(policy_loader=load_policy)
        else:
            module.main()
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
        os.chdir(previous_cwd)
