#!/usr/bin/env python3
"""Run the shared pi0_fast benchmark with constructed inputs."""
from _engine_benchmark import run

if __name__ == "__main__":
    run("pi0_fast", policy=True)
