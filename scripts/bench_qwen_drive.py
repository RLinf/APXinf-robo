#!/usr/bin/env python3
"""Run the shared qwen_drive benchmark with constructed inputs."""
from _engine_benchmark import run

if __name__ == "__main__":
    run("qwen_drive", policy=True)
