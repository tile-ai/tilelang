"""Restore the configure-time Windows SDK environment for a native build command."""

import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    selected = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    # Windows environment keys are case-insensitive, Python dictionaries are not.
    environment = {key: value for key, value in os.environ.items() if key.upper() not in selected}
    environment.update(selected)
    return subprocess.call(sys.argv[2:], env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
