from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Iterable


def _selected_env(keys: Iterable[str]) -> dict[str, str]:
    return {key: os.environ[key] for key in keys if key in os.environ}


def collect_env_report(extra_env_keys: Iterable[str] = ()) -> dict[str, object]:
    from tilelang import env as tilelang_env
    from tilelang.backend.target import list_target_detectors

    env_keys = [
        "CUDA_HOME",
        "CUDA_PATH",
        "ROCM_HOME",
        "ROCM_PATH",
        "TILELANG_DEFAULT_TARGET",
        "TILELANG_EXECUTION_BACKEND",
        "TILELANG_VERBOSE",
        "TILELANG_CACHE_DIR",
        "TILELANG_TMP_DIR",
        *extra_env_keys,
    ]
    return {
        "python": sys.version.replace("\n", ""),
        "platform": platform.platform(),
        "tilelang": {
            "cuda_home": tilelang_env.CUDA_HOME,
            "rocm_home": tilelang_env.ROCM_HOME,
            "default_target": tilelang_env.get_default_target(),
            "default_execution_backend": tilelang_env.get_default_execution_backend(),
            "verbose": tilelang_env.get_default_verbose(),
            "target_detectors": list(list_target_detectors()),
        },
        "environment": _selected_env(env_keys),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print a JSON TileLang environment report.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Additional environment variable key to include. Can be passed multiple times.",
    )
    args = parser.parse_args(argv)
    print(json.dumps(collect_env_report(args.env), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
