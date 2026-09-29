"""Load a benchmark from the pinned engine without duplicating its workload."""
from pathlib import Path
import importlib.util
import sys


def run(name, *, policy=False):
    root = Path(__file__).resolve().parents[1]
    engine = root / "apxinf"
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(engine / "scripts"))
    sys.path.insert(0, str(engine / "python/apxinf"))
    path = engine / "scripts" / f"bench_{name}.py"
    spec = importlib.util.spec_from_file_location(f"apxinf_bench_{name}", path)
    if not path.is_file() or spec is None or spec.loader is None:
        raise RuntimeError(f"missing engine benchmark: {path}; initialize the apxinf submodule")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if policy:
        from apxinf_robo import load_policy
        module.main(policy_loader=load_policy)
    else:
        module.main()
