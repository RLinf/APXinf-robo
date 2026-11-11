"""``apx-drobotics``: check a deployment against the contract."""

from __future__ import annotations

import sys

from .contract import main

if __name__ == "__main__":
    sys.exit(main())
