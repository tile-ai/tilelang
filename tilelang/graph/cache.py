"""Graph-level compilation cache for the TileLang torch.compile backend.

Three cache layers:
  - In-memory: graph hash → compiled wrapper (instant, same process)
  - Disk: graph hash → serialized Relax IRModule (avoids re-running pipeline)
  - Kernel-level: tilelang.compile() caches compiled binaries on disk
"""

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

import torch
from torch import fx

from tilelang import tvm as tvm
from tilelang import env as tilelang_env

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(
    os.environ.get(
        "TILELANG_GRAPH_CACHE_DIR",
        os.path.join(tilelang_env.TILELANG_CACHE_DIR, "graph_cache"),
    )
)

# In-memory cache: graph hash → wrapper callable
_memory_cache: dict[str, callable] = {}


def _hash_graph(gm: fx.GraphModule, example_inputs: list, compile_options=()) -> str:
    """Compute a hash key from the FX graph structure and input metadata."""
    h = hashlib.sha256()
    h.update(gm.graph.python_code("self").src.encode())

    for inp in example_inputs:
        if isinstance(inp, torch.Tensor):
            shape_str = str(tuple(str(dim) for dim in inp.shape))
            h.update(f"tensor:{shape_str}:{inp.dtype}:{inp.device}:{inp.layout}:{inp.requires_grad}".encode())
        elif isinstance(inp, (bool, int, float, str, type(None), torch.SymInt)):
            h.update(f"scalar:{type(inp).__name__}:{inp!s}".encode())
        else:
            h.update(f"other:{type(inp).__name__}".encode())

    h.update(repr(tuple(compile_options)).encode())

    return h.hexdigest()


def _disk_path(key: str) -> Path:
    return _CACHE_DIR / key[:2] / key


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------


def graph_cache_key(gm: fx.GraphModule, example_inputs: list, compile_options=()) -> str:
    """Compute cache key once; pass to other cache functions to avoid rehashing."""
    return _hash_graph(gm, example_inputs, compile_options)


def get_memory_cached(key: str):
    """Returns wrapper callable or None."""
    result = _memory_cache.get(key)
    if result is not None:
        logger.debug("Graph memory cache hit: %s", key[:12])
    return result


def put_memory_cached(key: str, wrapper: callable) -> None:
    _memory_cache[key] = wrapper


# ---------------------------------------------------------------------------
# Disk cache — saves/loads the optimized Relax IRModule
# ---------------------------------------------------------------------------


def save_relax_mod(key: str, mod: tvm.IRModule) -> None:
    """Save the optimized Relax IRModule to disk."""
    dest = _disk_path(key)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=dest.parent))
        mod_path = tmp / "relax_mod.json"
        with open(mod_path, "w") as f:
            f.write(tvm.ir.save_json(mod))
        if not dest.exists():
            tmp.rename(dest)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
        logger.debug("Graph disk cache saved: %s", key[:12])
    except Exception:
        logger.debug("Failed to save graph disk cache", exc_info=True)


def load_relax_mod(key: str):
    """Load the optimized Relax IRModule from disk. Returns IRModule or None."""
    mod_path = _disk_path(key) / "relax_mod.json"
    if not mod_path.exists():
        return None
    try:
        with open(mod_path) as f:
            mod = tvm.ir.load_json(f.read())
        logger.debug("Graph disk cache hit: %s", key[:12])
        return mod
    except Exception:
        logger.debug("Failed to load graph disk cache", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def clear_cache() -> None:
    """Clear both in-memory and disk graph caches."""
    _memory_cache.clear()
    if _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR, ignore_errors=True)
