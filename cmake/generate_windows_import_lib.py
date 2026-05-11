from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


EXPORT_RE = re.compile(r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(\S+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a .def file from a DLL's export table using dumpbin.")
    parser.add_argument("--dumpbin", required=True)
    parser.add_argument("--dll", required=True)
    parser.add_argument("--def", dest="def_file", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dll_path = pathlib.Path(args.dll)
    def_path = pathlib.Path(args.def_file)

    proc = subprocess.run(
        [args.dumpbin, "/exports", str(dll_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        if proc.stderr:
            print(f"dumpbin stderr: {proc.stderr}", file=sys.stderr)
        return proc.returncode

    exports: list[str] = []
    seen: set[str] = set()
    for line in proc.stdout.splitlines():
        match = EXPORT_RE.match(line)
        if not match:
            continue
        symbol = match.group(1)
        if symbol in seen:
            continue
        seen.add(symbol)
        exports.append(symbol)

    if not exports:
        print(f"No exports found in {dll_path}", file=sys.stderr)
        return 1

    def_path.write_text(
        "LIBRARY " + dll_path.name + "\nEXPORTS\n" + "\n".join(f"    {symbol}" for symbol in exports) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
