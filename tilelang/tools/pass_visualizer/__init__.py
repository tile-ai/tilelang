"""Interactive pass-by-pass IR structure-tree visualizer for TileLang kernels.

See ``viewer.py`` for the CLI entry point and ``README.md`` for usage. This is a
debugging complement to ``tilelang.utils.pass_diff``: where pass_diff shows a
text-level diff of the TVMScript, this tool renders the SBlock structure tree and
expands tile ops by field name, with per-class operator highlighting.
"""

from .core import (
    build_pass_stages,  # noqa: F401
    build_module,  # noqa: F401
    inspect_structure,  # noqa: F401
    load_user_module,  # noqa: F401
    discover_jit_kernels,  # noqa: F401
    kernel_to_tir,  # noqa: F401
)

# NOTE: viewer (build_pass_data / emit_html / emit_txt) is intentionally NOT
# imported here. Pre-importing it would trigger a RuntimeWarning when the module
# is run as ``python -m tilelang.tools.pass_visualizer.viewer``. Import those
# from ``tilelang.tools.pass_visualizer.viewer`` directly.
