from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tvm.target import Target

from tilelang.env import TL_BINS
from tilelang.jit.adapter.libgen import LibraryGenerator


SUNMMIO_TOOLCHAIN_ENV = "SUNMMIO_TOOLCHAIN"
SUNMMIO_SUDECK_ROOT_ENV = "SUNMMIO_SUDECK_ROOT"
SUNMMIO_SUDECK_MESH = "4x4"

SUNMMIO_CLANG_CFLAGS = (
    "--target=riscv64-sunmmio-elf",
    "-mno-relax",
    "-O2",
)
SUNMMIO_LDFLAGS = (
    "-fuse-ld=lld",
    "-Wl,--no-warn-mismatch,--gc-sections",
    "-lnosys",
)
SUNMMIO_SUDECK_LDFLAGS = (
    "-nostartfiles",
    "-fuse-ld=lld",
    "-Wl,--no-warn-mismatch,--emit-relocs",
    "-lnosys",
)
_SUNSIM_PWLN_TABLE_WORDS = (
    0x3FB551D5,
    0x3FBD580D,
    0x3FC5BA53,
    0x3FCE7B54,
    0x3FD79FA0,
    0x3FE12B85,
    0x3FEB239E,
    0x3FF58CB4,
    0x400035E4,
    0x4005E30B,
    0x400BD086,
    0x40120137,
    0x401877F6,
    0x401F385F,
    0x402643CD,
    0x402DA4B2,
    0x3F1547C6,
    0x3F043A8F,
    0x3EE2BAE1,
    0x3EB9261F,
    0x3E8B70A0,
    0x3E32A462,
    0x3D89F69D,
    0xBD4AF4CE,
    0xBE35322A,
    0xBEA18D78,
    0xBEEE9CB5,
    0xBF211709,
    0xBF4E5641,
    0xBF7F493C,
    0xBF9A0F79,
    0xBFB6A6F0,
    0xBF70EC02,
    0xBF56282E,
    0xBF3FA069,
    0xBF2C7082,
    0xBF1C00DC,
    0xBF0DEAFD,
    0xBF01856E,
    0xBEED78C8,
    0xBEDA7BE2,
    0xBEC9AB51,
    0xBEBAB44F,
    0xBEAD693F,
    0xBEA181B3,
    0xBE96A8CA,
    0xBE8CFEFE,
    0xBE83CCC7,
    0x3FF8675B,
    0x3FEA3186,
    0x3FDD86F4,
    0x3FD22403,
    0x3FC7DF3B,
    0x3FBEA220,
    0x3FB61D4E,
    0x3FAE5D82,
    0x3FA73F79,
    0x3FA0AE8D,
    0x3F9A9A9A,
    0x3F94FF70,
    0x3F8FCABC,
    0x3F8AE0D9,
    0x3F8659A7,
    0x3F81E549,
    0xBEF46B51,
    0xBEDFA893,
    0xBECDD108,
    0xBEBE3051,
    0xBEB073F6,
    0xBEA44A9E,
    0xBE99774C,
    0xBE8FC15F,
    0xBE870D20,
    0xBE7E84B3,
    0xBE70403D,
    0xBE6325BA,
    0xBE571825,
    0xBE4BFD2D,
    0xBE41BD2C,
    0xBE384294,
    0x3FBD123B,
    0x3FB78BD1,
    0x3FB27F8F,
    0x3FADE03B,
    0x3FA9A059,
    0x3FA5B48D,
    0x3FA21204,
    0x3F9EB15A,
    0x3F9B8B97,
    0x3F989AF8,
    0x3F95DA3F,
    0x3F93450A,
    0x3F90D806,
    0x3F8E90A0,
    0x3F8C6C5B,
    0x3F8A68ED,
    0x40784DF0,
    0x406A0FCD,
    0x405D5FCD,
    0x405208CD,
    0x4047DDCD,
    0x403EB98D,
    0x40367D8D,
    0x402F0B8D,
    0x40284F8D,
    0x4022355D,
    0x401CAD5D,
    0x4017A8DD,
    0x40131A3D,
    0x400EF62D,
    0x400B316D,
    0x4007C20D,
    0xC016C5D8,
    0xC012F74C,
    0xC00F6948,
    0xC00C1190,
    0xC008E7B0,
    0xC005E510,
    0xC003049C,
    0xC0004380,
    0xBFFA0758,
    0xBFF5180C,
    0xBFF05898,
    0xBFEBC0E8,
    0xBFE74E58,
    0xBFE2FF7C,
    0xBFDED288,
    0xBFDAC634,
)
_C_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PathLike = str | os.PathLike[str]

SUNMMIO_KERNEL_ELF_FILE = "kernel.elf"
SUNMMIO_KERNEL_OBJ_FILE = "kernel.o"
SUNMMIO_KERNEL_TIR_FILE = "kernel.tir"
SUNMMIO_KERNEL_MLIR_FILE = "kernel.mlir"
SUNMMIO_KERNEL_LLVM_IR_FILE = "kernel.ll"
SUNMMIO_SUDECK_LAUNCHER_CPP_FILE = "sudeck_launcher.cpp"
SUNMMIO_SUDECK_LAUNCHER_LIB_FILE = "sudeck_launcher.so"
SUNMMIO_SUDECK_LAUNCH_MODULE_FILE = "sudeck_launch.py"
SUNMMIO_SUDECK_LAUNCHER_CMAKE_BUILD_DIR = "sudeck_launcher_cmake"


@dataclass(frozen=True)
class SunmmioKernelArtifact:
    elf_path: Path
    mlir_path: Path | None
    llvm_ir_path: Path
    build_dir: Path
    runtime_kernel_name: str
    tir_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "elf_path", Path(self.elf_path))
        object.__setattr__(self, "mlir_path", None if self.mlir_path is None else Path(self.mlir_path))
        object.__setattr__(self, "llvm_ir_path", Path(self.llvm_ir_path))
        object.__setattr__(self, "build_dir", Path(self.build_dir))
        object.__setattr__(self, "tir_path", None if self.tir_path is None else Path(self.tir_path))


def _find_existing_path(
    description: str,
    candidates: Sequence[Path],
    *,
    executable: bool = False,
) -> Path:
    for candidate in candidates:
        if candidate.exists() and (not executable or os.access(candidate, os.X_OK)):
            return candidate

    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{description} not found. Checked:\n{checked}")


def find_npuir_tool(name: str) -> Path:
    candidates = [Path(bin_dir) / name for bin_dir in TL_BINS]

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    checked = "\n".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"{name} executable not found in TileLang binary directories. Rebuild TileLang with USE_SUNMMIO=ON "
        f"or install a Sunmmio-enabled TileLang package that includes NPU-IR tools. Checked:\n{checked}"
    )


@dataclass(frozen=True)
class NpuirTools:
    compile: Path

    @classmethod
    def resolve(cls) -> NpuirTools:
        return cls(compile=find_npuir_tool("npuir-compile"))


@dataclass(frozen=True)
class SunmmioToolchain:
    root: Path
    clangxx: Path

    @classmethod
    def resolve(cls) -> SunmmioToolchain:
        toolchain_env = os.getenv(SUNMMIO_TOOLCHAIN_ENV)
        if not toolchain_env:
            raise FileNotFoundError(f"{SUNMMIO_TOOLCHAIN_ENV} is not set. Set it to the Sunmmio toolchain root.")

        toolchain_path = _find_existing_path("Sunmmio toolchain", [Path(toolchain_env)])
        return cls(
            root=toolchain_path,
            clangxx=_find_existing_path(
                "Sunmmio clang++",
                [toolchain_path / "clang" / "bin" / "clang++"],
                executable=True,
            ),
        )

    def cflags(self) -> list[str]:
        return list(SUNMMIO_CLANG_CFLAGS)

    def resolve_device_ld(self, mcpu: str) -> Path:
        patterns = (
            f"sysroot/*/lib/sunmmio/{mcpu}/device.ld",
            f"clang/lib/clang-runtimes/*/lib/sunmmio/{mcpu}/device.ld",
        )
        for pattern in patterns:
            matches = sorted(self.root.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"Sunmmio device.ld for mcpu={mcpu} not found under {self.root}")


def _target_mcpu(target: Target) -> str:
    mcpu = target.attrs.get("mcpu") if target.attrs is not None else None
    if mcpu is None:
        raise ValueError(f"Sunmmio target is missing required `mcpu` attribute: {target}")
    return str(mcpu)


# FIXME: Remove this SuDeck-only linker workaround after SuBase stops packing fixed NOLOAD DTCM sections as movable kernel DTCM.
_RUNNER_OWNED_SECTIONS = (
    (r"[ \t]*\.dtcm\.scratch\s*\(NOLOAD\)\s*:\s*\{.*?\}\s*>\s*DTCM_SCRATCH\s*", "DTCM_SCRATCH"),
    (r"[ \t]*\.dtcm\.tagtrace\s*\(NOLOAD\)\s*:\s*\{.*?\}\s*>\s*DTCM_TAGTRACE\s*", "DTCM_TAGTRACE"),
    (r"[ \t]*\.stack\s*\(NOLOAD\)\s*:\s*\{.*?\}\s*>\s*STACK\s*", "STACK"),
)
_RUNNER_OWNED_SYMBOLS = (
    ("_dtcm_scratch_start", "DTCM_SCRATCH_START"),
    ("_dtcm_scratch_end", "DTCM_SCRATCH_START + DTCM_SCRATCH_SIZE"),
    ("_dtcm_tagtrace_start", "DTCM_TAGTRACE_START"),
    ("_dtcm_tagtrace_end", "DTCM_TAGTRACE_START + DTCM_TAGTRACE_SIZE"),
)


def _write_sudeck_linker_script(toolchain: SunmmioToolchain, mcpu: str, out_path: Path) -> Path:
    """Derive a runner-compatible linker script by dropping the runner-owned
    DTCM sections while preserving their compiler-visible address symbols."""
    text = toolchain.resolve_device_ld(mcpu).read_text(encoding="utf-8")
    for pattern, region in _RUNNER_OWNED_SECTIONS:
        text, count = re.subn(pattern, "\n", text, flags=re.DOTALL)
        if count == 0:
            raise RuntimeError(f"device.ld for mcpu={mcpu} is missing runner-owned section {region}")
    # _stack/_stack_end no longer exist; point the provided symbols at the runner's stack region.
    text = re.sub(r"PROVIDE\(__stack_start\s*=\s*_stack\);", "PROVIDE(__stack_start = DTCM_STACK_START);", text)
    text = re.sub(r"PROVIDE\(__stack_end\s*=\s*_stack_end\);", "PROVIDE(__stack_end = DTCM_STACK_START + DTCM_STACK_SIZE);", text)
    symbol_provides = "\n".join(f"PROVIDE({name} = {value});" for name, value in _RUNNER_OWNED_SYMBOLS)
    text = f"{text.rstrip()}\n{symbol_provides}\n"
    out_path.write_text(text, encoding="utf-8")
    return out_path


@dataclass(frozen=True)
class SuDeckToolchain:
    root: Path
    library_path: Path

    @classmethod
    def resolve(cls) -> SuDeckToolchain:
        root_env = os.getenv(SUNMMIO_SUDECK_ROOT_ENV)
        if not root_env:
            raise FileNotFoundError(f"{SUNMMIO_SUDECK_ROOT_ENV} is not set. Set it to the SuDeck install root.")

        root = _find_existing_path("SuDeck install root", [Path(root_env).expanduser()])
        cmake_dir = root / "lib" / "cmake" / "SuDeck"
        _find_existing_path("SuDeck CMake package", [cmake_dir / "SuDeckConfig.cmake"])
        _find_existing_path("SuDeck CMake target", [cmake_dir / f"SuDeckTargets_{SUNMMIO_SUDECK_MESH}.cmake"])
        # FIXME: Drop the SUNMMIO_SUDECK_MESH when we have the symlink `libsudeck.so`
        library_path = _find_existing_path("SuDeck library", [root / "lib" / f"libsudeck_{SUNMMIO_SUDECK_MESH}.so"])
        return cls(root=root.resolve(), library_path=library_path.resolve())

    @property
    def library_dir(self) -> Path:
        return self.library_path.parent

    @property
    def cmake_component(self) -> str:
        return f"mesh_{SUNMMIO_SUDECK_MESH}"

    @property
    def cmake_target(self) -> str:
        return f"sudeck::sudeck_{SUNMMIO_SUDECK_MESH}"

    def cmake_prefix_path(self) -> str:
        prefixes = [str(self.root)]
        env_prefix = os.getenv("CMAKE_PREFIX_PATH")
        if env_prefix:
            prefixes.extend(prefix for prefix in env_prefix.split(os.pathsep) if prefix)
        return ";".join(prefixes)


def _tvm_ffi_paths() -> tuple[Path, Path]:
    """Return (include_dir, lib_path) for building the launcher against tvm-ffi."""
    import tvm_ffi.libinfo as libinfo

    return Path(libinfo.include_paths()[0]), Path(libinfo.find_libtvm_ffi())


def _cmake_quote(value: Path | str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _write_sudeck_launcher_cmake(
    build_dir: Path,
    sudeck: SuDeckToolchain,
    ffi_include: Path,
    ffi_library: Path,
) -> Path:
    cmake_path = build_dir / "CMakeLists.txt"
    rpath = f"{ffi_library.parent};{sudeck.library_dir}"
    cmake_path.write_text(
        f"""\
cmake_minimum_required(VERSION 3.22)
project(TileLangSunmmioSuDeckLauncher LANGUAGES CXX)

find_package(SuDeck REQUIRED COMPONENTS {sudeck.cmake_component})

add_library(tilelang_sudeck_launcher SHARED "{SUNMMIO_SUDECK_LAUNCHER_CPP_FILE}")
target_compile_features(tilelang_sudeck_launcher PRIVATE cxx_std_17)
target_include_directories(tilelang_sudeck_launcher PRIVATE "{_cmake_quote(ffi_include)}")
target_link_libraries(tilelang_sudeck_launcher PRIVATE "{_cmake_quote(ffi_library)}" {sudeck.cmake_target})
set_target_properties(tilelang_sudeck_launcher PROPERTIES
  PREFIX ""
  OUTPUT_NAME "sudeck_launcher"
  LIBRARY_OUTPUT_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
  RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_CURRENT_SOURCE_DIR}}"
  BUILD_RPATH "{_cmake_quote(rpath)}"
  INSTALL_RPATH "{_cmake_quote(rpath)}"
)
""",
        encoding="utf-8",
    )
    return cmake_path


def _run_command(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    description: str,
    input_text: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd is not None else None,
        input=input_text,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{description} failed\ncommand: {' '.join(str(part) for part in command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _render_sunsim_pwln_setup() -> str:
    table_rows = []
    for offset in range(0, len(_SUNSIM_PWLN_TABLE_WORDS), 8):
        row = ", ".join(f"0x{word:08X}" for word in _SUNSIM_PWLN_TABLE_WORDS[offset : offset + 8])
        table_rows.append(f"    {row},")
    table = "\n".join(table_rows)
    return f"""\
__attribute__((section(".r.sram.common"))) static const uint32_t pwln_tables[{len(_SUNSIM_PWLN_TABLE_WORDS)}] = {{
{table}
}};

__device static inline void pwln_init() {{
  __rsram float *base = (__rsram float *)pwln_tables;
  auto t_exp = su_vload(base);
  auto t_recip = su_vload(base + 32);
  auto t_rsqrt = su_vload(base + 64);
  auto t_ln = su_vload(base + 96);
  (void)su_vfpwln_set_exp(t_exp);
  (void)su_vfpwln_set_recip(t_recip);
  (void)su_vfpwln_set_rsqrt(t_rsqrt);
  (void)su_vfpwln_set_ln(t_ln);
  su_sync_vector();
}}
"""


def _render_sunsim_main_thunk(kernel_name: str) -> str:
    if not _C_IDENTIFIER_RE.match(kernel_name):
        raise ValueError(f"Sunmmio kernel name must be a C identifier for sunsim main thunk generation: {kernel_name!r}")

    return f"""\
#include <stddef.h>
#include <stdint.h>

#include <sunmmio_dev.h>

{_render_sunsim_pwln_setup()}

extern "C" {{

__attribute__((used, section(".kernargs"), aligned(8))) volatile unsigned char _kernel_arg_start[4096] = {{1}};
__attribute__((used, section(".descriptors"), aligned(8))) volatile unsigned char _descriptor_start[4096] = {{1}};

void {kernel_name}(void *args);

int main(void) {{
  pwln_init();
  {kernel_name}(const_cast<unsigned char *>(_kernel_arg_start));
  return 0;
}}

}}  // extern "C"
    """


class SunmmioLibraryGenerator(LibraryGenerator):
    """Generate Sunmmio library artifacts from TileLang SUVM MLIR."""

    def __init__(self, target: Target, verbose: bool = False):
        super().__init__(target, verbose)
        self.mlir_source: str = ""
        self.device_tir_source: str = ""
        self.artifact: SunmmioKernelArtifact | None = None

    def update_mlir_source(self, mlir_source: str):
        self.mlir_source = mlir_source

    def update_device_tir_source(self, tir_source: str | None):
        self.device_tir_source = tir_source or ""

    def _compile_flags(self, toolchain: SunmmioToolchain) -> list[str]:
        cflags = toolchain.cflags()
        cflags.append(f"-mcpu={_target_mcpu(self.target)}")
        return cflags

    def load_lib(self, lib_path: PathLike | None = None):
        if lib_path is None:
            if self.libpath is None:
                raise RuntimeError("SunmmioLibraryGenerator.libpath is not set; call compile_lib() first or pass lib_path explicitly.")
            lib_path = self.libpath

        elf_path = Path(lib_path)
        if elf_path.suffix != ".elf":
            raise ValueError(f"Sunmmio kernel artifact must be an ELF file, got: {elf_path}")
        mlir_path = elf_path.parent / SUNMMIO_KERNEL_MLIR_FILE
        if not mlir_path.exists():
            raise RuntimeError(f"Cached Sunmmio ELF is missing sibling MLIR artifact: {elf_path}")
        llvm_ir_path = elf_path.parent / SUNMMIO_KERNEL_LLVM_IR_FILE
        if not llvm_ir_path.exists():
            raise RuntimeError(f"Cached Sunmmio ELF is missing sibling LLVM IR artifact: {elf_path}")
        tir_path = elf_path.parent / SUNMMIO_KERNEL_TIR_FILE

        runtime_kernel_name = getattr(self, "runtime_kernel_name", "kernel")
        self.artifact = SunmmioKernelArtifact(
            elf_path=elf_path,
            mlir_path=mlir_path,
            llvm_ir_path=llvm_ir_path,
            build_dir=elf_path.parent,
            runtime_kernel_name=runtime_kernel_name,
            tir_path=tir_path if tir_path.exists() else None,
        )
        self.srcpath = str(llvm_ir_path)
        self.libpath = str(elf_path)

    def compile_lib(self, timeout: float | None = None):
        raise NotImplementedError("SunmmioLibraryGenerator only lowers to LLVM IR; use a runtime-specific generator.")

    def _dump_mlir(self, mlir_path: Path) -> None:
        if not self.mlir_source.strip():
            raise ValueError("Sunmmio kernel has no SUVM MLIR source to lower")
        mlir_path.write_text(self.mlir_source, encoding="utf-8")

    def _dump_device_tir(self, tir_path: Path) -> Path | None:
        if not self.device_tir_source.strip():
            return None
        tir_path.write_text(self.device_tir_source, encoding="utf-8")
        return tir_path

    def _mlir_to_llvm_ir(self, mlir_path: Path, llvm_path: Path) -> None:
        tools = NpuirTools.resolve()
        self._run_npuir_compile(tools, mlir_path, llvm_path)

    def _compile_kernel_obj(
        self,
        toolchain: SunmmioToolchain,
        llvm_path: Path,
        build_dir: Path,
        timeout: float | None,
    ) -> Path:
        kernel_obj = build_dir / SUNMMIO_KERNEL_OBJ_FILE
        _run_command(
            [toolchain.clangxx, "-c", *self._compile_flags(toolchain), "-o", kernel_obj, llvm_path],
            description="Sunmmio kernel LLVM IR compilation",
            timeout=timeout,
        )
        return kernel_obj

    def _run_npuir_compile(self, tools: NpuirTools, input_path: Path, output_path: Path) -> None:
        command = [
            str(tools.compile),
            f"--target={_target_mcpu(self.target)}",
            "--emit=llvm-ir",
            str(input_path),
            "-o",
            str(output_path),
        ]
        _run_command(
            command,
            description=(f"npuir-compile failed while lowering Sunmmio SUVM MLIR\nbuild_dir: {input_path.parent}\nmlir: {input_path}"),
        )


class SunmmioSuDeckLibraryGenerator(SunmmioLibraryGenerator):
    """Build a SuDeck-loadable Sunmmio kernel ELF plus its tvm-ffi launch module."""

    def __init__(self, target: Target, kernel_name: str, verbose: bool = False):
        super().__init__(target, verbose)
        self.runtime_kernel_name = kernel_name
        self.launcher_specs: list = []
        self.pymodule = None

    def update_launcher_specs(self, specs) -> None:
        """Device-ABI order params: (name, "tensor") or (name, c_scalar_type)."""
        self.launcher_specs = list(specs)

    def compile_lib(
        self,
        timeout: float | None = None,
        output_dir: PathLike | None = None,
    ):
        build_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="tilelang-sunmmio-sudeck-"))
        build_dir.mkdir(parents=True, exist_ok=True)

        elf_path = build_dir / SUNMMIO_KERNEL_ELF_FILE
        llvm_path = build_dir / SUNMMIO_KERNEL_LLVM_IR_FILE
        mlir_path = build_dir / SUNMMIO_KERNEL_MLIR_FILE
        tir_path = build_dir / SUNMMIO_KERNEL_TIR_FILE

        dumped_tir_path = self._dump_device_tir(tir_path)
        self._dump_mlir(mlir_path)
        self._build_kernel(build_dir, mlir_path, llvm_path, elf_path, timeout)
        self._build_host_launch_module(build_dir, elf_path, timeout)

        self.artifact = SunmmioKernelArtifact(
            runtime_kernel_name=self.runtime_kernel_name,
            build_dir=build_dir,
            elf_path=elf_path,
            llvm_ir_path=llvm_path,
            mlir_path=mlir_path,
            tir_path=dumped_tir_path,
        )
        self.srcpath = str(llvm_path)
        self.libpath = str(elf_path)

    def _build_kernel(self, build_dir: Path, mlir_path: Path, llvm_path: Path, elf_path: Path, timeout: float | None) -> None:
        sunmmio_toolchain = SunmmioToolchain.resolve()
        self._mlir_to_llvm_ir(mlir_path, llvm_path)
        kernel_obj = self._compile_kernel_obj(sunmmio_toolchain, llvm_path, build_dir, timeout)
        linker_script = _write_sudeck_linker_script(sunmmio_toolchain, _target_mcpu(self.target), build_dir / "device_sudeck.ld")
        _run_command(
            [
                sunmmio_toolchain.clangxx,
                *self._compile_flags(sunmmio_toolchain),
                kernel_obj,
                *SUNMMIO_SUDECK_LDFLAGS,
                "-T",
                str(linker_script),
                "-o",
                elf_path,
            ],
            description="Sunmmio SuDeck kernel ELF link",
            timeout=timeout,
        )

    def _build_host_launch_module(self, build_dir: Path, elf_path: Path, timeout: float | None) -> None:
        from .host_wrapper import SunmmioSuDeckSourceWrapper

        wrapper = SunmmioSuDeckSourceWrapper(
            self.launcher_specs,
            self.runtime_kernel_name,
            elf_path.name,
            SUNMMIO_SUDECK_LAUNCHER_LIB_FILE,
        )
        src_path = build_dir / SUNMMIO_SUDECK_LAUNCHER_CPP_FILE
        src_path.write_text(wrapper.generate_launcher_cpp(), encoding="utf-8")

        sudeck = SuDeckToolchain.resolve()
        ffi_include, ffi_library = _tvm_ffi_paths()
        _write_sudeck_launcher_cmake(build_dir, sudeck, ffi_include, ffi_library)
        cmake_build_dir = build_dir / SUNMMIO_SUDECK_LAUNCHER_CMAKE_BUILD_DIR
        _run_command(
            [
                "cmake",
                "-S",
                build_dir,
                "-B",
                cmake_build_dir,
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DCMAKE_PREFIX_PATH={sudeck.cmake_prefix_path()}",
            ],
            description="Sunmmio SuDeck launcher CMake configure",
            timeout=timeout,
        )
        _run_command(
            [
                "cmake",
                "--build",
                cmake_build_dir,
                "--target",
                "tilelang_sudeck_launcher",
                "--config",
                "Release",
            ],
            description="Sunmmio SuDeck launcher CMake build",
            timeout=timeout,
        )
        (build_dir / SUNMMIO_SUDECK_LAUNCH_MODULE_FILE).write_text(wrapper.generate_python_module(), encoding="utf-8")

    def load_lib(self, lib_path: PathLike | None = None):
        super().load_lib(lib_path)
        self._import_pymodule(self.artifact.build_dir)

    def _import_pymodule(self, build_dir: Path) -> None:
        import importlib.util

        py_path = build_dir / SUNMMIO_SUDECK_LAUNCH_MODULE_FILE
        if not py_path.exists():
            raise RuntimeError(f"Sunmmio SuDeck launch module missing: {py_path}")
        spec = importlib.util.spec_from_file_location(f"sunmmio_sudeck_launch_{self.runtime_kernel_name}", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.pymodule = module


class SunmmioSunsimLibraryGenerator(SunmmioLibraryGenerator):
    """Build a sunsim-loadable ELF from Sunmmio LLVM IR."""

    def __init__(self, target: Target, kernel_name: str, verbose: bool = False):
        super().__init__(target, verbose)
        self.runtime_kernel_name = kernel_name

    def compile_lib(
        self,
        timeout: float | None = None,
        output_dir: PathLike | None = None,
    ):
        build_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="tilelang-sunmmio-sunsim-"))
        build_dir.mkdir(parents=True, exist_ok=True)
        sunmmio_toolchain = SunmmioToolchain.resolve()

        mlir_path = build_dir / SUNMMIO_KERNEL_MLIR_FILE
        llvm_path = build_dir / SUNMMIO_KERNEL_LLVM_IR_FILE
        tir_path = build_dir / SUNMMIO_KERNEL_TIR_FILE

        dumped_tir_path = self._dump_device_tir(tir_path)
        self._dump_mlir(mlir_path)
        self._mlir_to_llvm_ir(mlir_path, llvm_path)

        thunk_path = self._write_main_thunk(build_dir)
        kernel_obj = self._compile_kernel_obj(sunmmio_toolchain, llvm_path, build_dir, timeout)
        thunk_obj = self._compile_thunk_obj(sunmmio_toolchain, thunk_path, build_dir, timeout)
        elf_path = self._link_elf(sunmmio_toolchain, kernel_obj, thunk_obj, build_dir, timeout)

        self.artifact = SunmmioKernelArtifact(
            elf_path=elf_path,
            mlir_path=mlir_path,
            llvm_ir_path=llvm_path,
            build_dir=build_dir,
            runtime_kernel_name=self.runtime_kernel_name,
            tir_path=dumped_tir_path,
        )
        self.srcpath = str(llvm_path)
        self.libpath = str(elf_path)

    def _write_main_thunk(self, build_dir: Path) -> Path:
        thunk_path = build_dir / "main_thunk.cpp"
        thunk_path.write_text(_render_sunsim_main_thunk(self.runtime_kernel_name), encoding="utf-8")
        return thunk_path

    def _compile_thunk_obj(
        self,
        toolchain: SunmmioToolchain,
        thunk_path: Path,
        build_dir: Path,
        timeout: float | None,
    ) -> Path:
        thunk_obj = build_dir / "main_thunk.o"
        _run_command(
            [
                toolchain.clangxx,
                "-c",
                *self._compile_flags(toolchain),
                "-x",
                "sunmmio",
                "-o",
                thunk_obj,
                thunk_path,
            ],
            description="Sunmmio sunsim main thunk compilation",
            timeout=timeout,
        )
        return thunk_obj

    def _link_elf(
        self,
        toolchain: SunmmioToolchain,
        kernel_obj: Path,
        thunk_obj: Path,
        build_dir: Path,
        timeout: float | None,
    ) -> Path:
        elf_path = build_dir / SUNMMIO_KERNEL_ELF_FILE
        _run_command(
            [toolchain.clangxx, *self._compile_flags(toolchain), kernel_obj, thunk_obj, *SUNMMIO_LDFLAGS, "-o", elf_path],
            description="Sunmmio sunsim ELF link",
            timeout=timeout,
        )
        return elf_path
