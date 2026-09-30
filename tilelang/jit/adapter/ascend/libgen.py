"""Library generation for the AscendC backend."""

from __future__ import annotations

import tempfile

from tilelang.env import TILELANG_TEMPLATE_PATH
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.transform import PassConfigKey


class AscendLibraryGenerator(LibraryGenerator):
    def compile_lib(self, timeout: float = None):
        from tilelang.contrib.bisheng import (
            find_bisheng_path,
            get_bisheng_compile_options,
            get_target_npu_arch,
            normalize_options,
        )

        src = tempfile.NamedTemporaryFile(mode="w", suffix=".asc", delete=False)  # noqa: SIM115
        libpath = src.name.replace(".asc", ".so")

        npu_arch = get_target_npu_arch(self.target)
        configured_options = normalize_options((self.pass_configs or {}).get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS))
        explicit_options = normalize_options(self.compile_flags)
        # Keep repeated option/value pairs such as -mllvm intact and ordered.
        command = [
            find_bisheng_path(),
            *get_bisheng_compile_options(npu_arch),
            *configured_options,
            *explicit_options,
            # Avoid using mmap to write linker output, thus more friendly for distributed FS
            "-Wl,--no-mmap-output-file",
            "--shared",
            src.name,
        ]
        command += ["-I" + TILELANG_TEMPLATE_PATH, "-o", libpath]
        self._run_compile(command, src, libpath, timeout)
