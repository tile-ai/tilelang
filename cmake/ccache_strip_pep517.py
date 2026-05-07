"""ccache prefix_command_cpp wrapper for PEP 517 build-isolation tempdir names.

uv (and pip) materialize each PEP 517 build into a freshly-named tempdir like
``<UV_CACHE_DIR>/builds-v0/.tmpXXXXXXXX``. That ``.tmpXXX`` segment shows up in
``-I`` flags (NVIDIA cu13, z3, ...) and is preserved inside cl.exe's
preprocessor output as ``#line N "...\\.tmpXXX\\..."`` directives. ccache hashes
the preprocessed text to derive its result key, so the random tempdir name
defeats cache hits across rebuilds.

This wrapper is plugged into ccache via ``prefix_command_cpp``. ccache invokes
it as::

    python ccache_strip_pep517.py <cl.exe path> <cl.exe args including /Fi...>

We forward the call to cl.exe unchanged, then post-process the file cl.exe
wrote to (the path given via ``/Fi`` or ``-Fi``) by replacing every
``.tmpXXXXXXXX`` token with a stable ``_pep517`` placeholder. ccache then sees
identical preprocessed bytes across rebuilds and the result-key lookup hits.

Note this only stabilizes ccache's *result key* (preprocessor mode). The
*manifest key* still embeds the raw command line, so direct-mode lookup keeps
missing on every rebuild -- but ccache transparently falls back to
preprocessor mode on direct miss, so net effect is one ``cl.exe -P`` run per
file (~10 ms) instead of a full ``-c`` compile.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

_TMP_PATTERN = re.compile(rb"\.tmp[A-Za-z0-9]+")
_STABLE_TOKEN = b"_pep517"


def _find_output_file(args: list[str]) -> str | None:
    for arg in args:
        if arg[:3] in ("/Fi", "-Fi") and len(arg) > 3:
            return arg[3:]
    return None


def _log(msg: str) -> None:
    log_path = os.environ.get("PEP517_STRIP_LOG")
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass


def main() -> int:
    cmd = sys.argv[1:]
    if not cmd:
        sys.stderr.write("ccache_strip_pep517: no compiler command supplied\n")
        return 2

    rc = subprocess.call(cmd)
    if rc != 0:
        _log(f"compile-failed rc={rc} src={cmd[-1]}")
        return rc

    out_path = _find_output_file(cmd)
    if not out_path:
        _log(f"no-output-arg src={cmd[-1]}")
        return 0
    if not os.path.isfile(out_path):
        _log(f"output-missing path={out_path}")
        return 0

    with open(out_path, "rb") as f:
        data = f.read()

    matches = _TMP_PATTERN.findall(data)
    # also probe for raw substrings to distinguish "no .tmp pattern" vs encoding issue
    has_uvcache_bytes = b"uv-cache" in data or b"uv\\cache" in data or b"uv/cache" in data
    has_tmp_bytes = b".tmp" in data
    head = data[:128].hex() if data else ""
    new_data = _TMP_PATTERN.sub(_STABLE_TOKEN, data)
    if new_data != data:
        with open(out_path, "wb") as f:
            f.write(new_data)
        _log(f"stripped count={len(matches)} src={cmd[-1]} out={out_path}")
    else:
        _log(f"no-tmp-found src={cmd[-1]} bytes={len(data)} has_uvcache={has_uvcache_bytes} has_tmp={has_tmp_bytes} head={head}")
        # Save a copy for forensic inspection
        keep = os.environ.get("PEP517_STRIP_KEEP")
        if keep:
            try:
                import shutil

                shutil.copy2(out_path, keep)
            except OSError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
