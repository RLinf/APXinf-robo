#!/usr/bin/env python3
"""Run the ApxInf pi0_fast benchmark through Robo's policy loader."""
from _engine_benchmark import run

if __name__ == "__main__":
    run("pi0_fast", policy=True)
