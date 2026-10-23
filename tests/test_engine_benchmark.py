"""Robo benchmarks must resolve the same default engine tactics as ApxInf."""

import importlib.util
import json
from pathlib import Path
import sys


def test_engine_benchmark_keeps_caller_paths_and_engine_tactics(tmp_path, monkeypatch):
    project = tmp_path / "robo"
    engine = project / "apxinf"
    scripts = engine / "scripts"
    scripts.mkdir(parents=True)
    default_tactics = engine / "configs/tuning/nvidia/orin-sm87/tactics.json"
    default_tactics.parent.mkdir(parents=True)
    default_tactics.write_text("{}")
    (scripts / "bench_gr00t.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "def main():\n"
        "    Path('probe.json').write_text(json.dumps({\n"
        "        'cwd': str(Path.cwd()), 'argv': sys.argv,\n"
        "        'default_tactics': Path('configs/tuning/nvidia/orin-sm87/tactics.json').is_file(),\n"
        "    }))\n"
    )
    wrapper = project / "scripts/_engine_benchmark.py"
    wrapper.parent.mkdir()
    spec = importlib.util.spec_from_file_location("engine_benchmark", Path(__file__).resolve().parents[1] / "scripts/_engine_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "__file__", str(wrapper))
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    argv = ["bench_gr00t.py", "--model-dir", "model", "--tactics=run/tactics.json",
            "--binary", "bin/gr00t_bench", "--frames", "run/frames.npz",
            "--out", "run/report.json"]
    monkeypatch.setattr(sys, "argv", argv)
    previous_path = sys.path[:]

    module.run("gr00t")

    probe = json.loads((engine / "probe.json").read_text())
    assert probe["cwd"] == str(engine)
    assert probe["default_tactics"] is True
    assert probe["argv"] == [
        "bench_gr00t.py", "--model-dir", str(caller / "model"),
        f"--tactics={caller / 'run/tactics.json'}", "--binary", str(caller / "bin/gr00t_bench"),
        "--frames", str(caller / "run/frames.npz"),
        "--out", str(caller / "run/report.json"),
    ]
    assert Path.cwd() == caller
    assert sys.argv is argv
    assert sys.path == previous_path
