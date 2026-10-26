"""Build or install TileLang's pinned, unmodified standalone TIRx dependency."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile


PROJECT = Path(__file__).resolve().parents[2]
EXTENSION_EXPORTS = PROJECT / "cmake" / "tirx_extension_exports.cmake"
BACKENDS = ("CUDA", "ROCM", "METAL", "LLVM", "OPENCL", "VULKAN", "Z3")
FALSE_VALUES = {"0", "off", "false", "no", ""}


def run(command, **kwargs):
    command = [str(argument) for argument in command]
    print("+ " + shlex.join(command), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def uv_command():
    executable = shutil.which("uv")
    if executable:
        return [executable]
    run([sys.executable, "-m", "pip", "install", "uv"])
    return [sys.executable, "-m", "uv"]


def project_metadata(path, uv):
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            run([*uv, "pip", "install", "--python", sys.executable, "tomli"])
            import tomli as tomllib
    return tomllib.loads(path.read_text())


def required_version(project=PROJECT):
    return (project / "TVM_TIRX_VERSION").read_text().strip()


def validate_revision(revision):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("TIRx revision must be a full, pinned 40-character Git commit")
    return revision.lower()


def validate_source(source, revision, project=PROJECT):
    requested = Path(os.path.abspath(source))
    source = requested.resolve()
    if requested.is_relative_to((project / "3rdparty").absolute()) or source.is_relative_to((project / "3rdparty").resolve()):
        raise ValueError("Standalone TIRx must not be placed under TileLang's 3rdparty directory")
    for relative in ("include/tvm/tirx/type.h", "python/tvm_tirx/__init__.py", "3rdparty/tvm-ffi/CMakeLists.txt"):
        if not (source / relative).is_file():
            raise ValueError(f"Incomplete standalone TIRx checkout: missing {relative}")
    if not (source / ".git").exists():
        raise ValueError("TIRx source must be a Git checkout so the pinned revision and unmodified source can be verified")
    actual = run(["git", "-C", source, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if actual != revision:
        raise ValueError(f"TIRx checkout is at {actual}, expected the upstream revision {revision}")
    dirty = run(
        ["git", "-C", source, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none"],
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise ValueError(f"TIRx source must be unmodified at {revision}:\n{dirty}")
    return source


def definitions_from_environment(environment, platform=sys.platform, command_line=()):
    defaults = {"CUDA": "AUTO", "ROCM": "OFF", "METAL": "ON" if platform == "darwin" else "OFF", "LLVM": "OFF", "Z3": "ON"}
    definitions = {"CMAKE_PROJECT_tvm_tirx_INCLUDE": str(EXTENSION_EXPORTS)}
    for backend in BACKENDS:
        definitions[f"TVM_TIRX_USE_{backend}"] = environment.get(
            f"TVM_TIRX_USE_{backend}", environment.get(f"USE_{backend}", defaults.get(backend, "OFF"))
        )
    for name in ("TVM_TIRX_ROCM_PATH", "CMAKE_CUDA_ARCHITECTURES", "CMAKE_CXX_COMPILER_LAUNCHER", "CMAKE_C_COMPILER_LAUNCHER"):
        if environment.get(name):
            definitions[name] = environment[name]
    overrides = json.loads(environment.get("TVM_TIRX_CMAKE_DEFINES", "{}"))
    if not isinstance(overrides, dict) or not all(
        isinstance(name, str) and isinstance(value, (str, int, bool)) for name, value in overrides.items()
    ):
        raise ValueError("TVM_TIRX_CMAKE_DEFINES must be a JSON object of CMake settings")
    definitions.update({name: "ON" if value is True else "OFF" if value is False else str(value) for name, value in overrides.items()})
    for setting in command_line:
        name, separator, value = setting.partition("=")
        if not separator or not name:
            raise ValueError("CMake definitions must use NAME=VALUE")
        definitions[name] = value
    if "TVM_TIRX_EXPORT_ALL_SYMBOLS" in definitions:
        raise ValueError("TVM_TIRX_EXPORT_ALL_SYMBOLS is obsolete; TileLang owns its extension export policy")
    if definitions["CMAKE_PROJECT_tvm_tirx_INCLUDE"] != str(EXTENSION_EXPORTS):
        raise ValueError("TileLang requires its CMAKE_PROJECT_tvm_tirx_INCLUDE export hook")
    return definitions


def validate_wheel(path, version):
    with zipfile.ZipFile(path) as archive:
        metadata_paths = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError(f"Expected one distribution in {path}")
        from email.parser import Parser

        metadata = Parser().parsestr(archive.read(metadata_paths[0]).decode())
        if re.sub(r"[-_.]+", "-", metadata["Name"]).lower() != "apache-tvm-tirx" or metadata["Version"] != version:
            raise ValueError(f"Expected apache-tvm-tirx=={version}, received {metadata['Name']}=={metadata['Version']}")
        if any(name.startswith("tvm/") or "3rdparty/tvm/" in name for name in archive.namelist()):
            raise ValueError("A standalone dependency wheel must not contain legacy TVM")


def installed_configuration(python):
    environment = {
        name: value for name, value in os.environ.items() if name not in {"PYTHONPATH", "TVM_TIRX_LIBRARY_PATH", "TVM_TIRX_SOURCE"}
    }
    result = run(
        [python, "-I", "-c", "import json; from tvm_tirx import config; print(json.dumps(config.configuration()))"],
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(result.stdout)


def validate_backends(configuration, definitions):
    directory = Path(configuration["library_dir"])
    for backend in ("CUDA", "ROCM", "METAL", "OPENCL", "VULKAN"):
        setting = definitions[f"TVM_TIRX_USE_{backend}"].lower()
        if setting in FALSE_VALUES or setting == "auto":
            continue
        if not any(directory.glob(f"*tvm_runtime_{backend.lower()}.*")):
            raise ValueError(f"TIRx package lacks the requested {backend} runtime; build that backend or provide a matching wheel")


def provision(arguments):
    version = required_version()
    pinned_revision = validate_revision((PROJECT / "TVM_TIRX_COMMIT").read_text().strip())
    revision = validate_revision(arguments.revision or pinned_revision)
    if revision != pinned_revision:
        raise ValueError("Update TVM_TIRX_COMMIT with the upstream revision; environment variables cannot override the source pin")
    output = arguments.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    uv = uv_command()
    definitions = definitions_from_environment(os.environ, command_line=arguments.define)
    tilelang_metadata = project_metadata(PROJECT / "pyproject.toml", uv)
    run(
        [
            *uv,
            "pip",
            "install",
            "--python",
            arguments.python,
            *tilelang_metadata["build-system"]["requires"],
            "cmake>=3.26.1",
            "ninja>=1.11",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="tilelang-tirx-") as temporary:
        temporary = Path(temporary)
        if arguments.wheel:
            arguments.wheel = arguments.wheel.format(platform=sys.platform, machine=platform.machine())
            parsed = urllib.parse.urlparse(arguments.wheel)
            if parsed.scheme in ("https", "http"):
                wheel = temporary / Path(urllib.parse.unquote(parsed.path)).name
                if wheel.suffix != ".whl":
                    raise ValueError("TVM_TIRX_WHEEL must identify a wheel file, not a package index")
                with urllib.request.urlopen(arguments.wheel) as response, wheel.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
            else:
                wheel = Path(arguments.wheel).resolve()
            validate_wheel(wheel, version)
        else:
            if arguments.source:
                source = Path(arguments.source)
            elif arguments.repository:
                source = temporary / "tvm-tirx"
                run(["git", "init", source])
                run(["git", "-C", source, "remote", "add", "origin", arguments.repository])
                run(["git", "-C", source, "fetch", "--depth", "1", "origin", revision])
                run(["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"])
                run(["git", "-C", source, "submodule", "update", "--init", "--recursive", "--depth", "1"])
            elif (PROJECT.parent / "tvm-tirx").is_dir():
                source = PROJECT.parent / "tvm-tirx"
            else:
                raise ValueError(
                    "Provide TVM_TIRX_SOURCE, TVM_TIRX_WHEEL, or TVM_TIRX_REPOSITORY for the pinned dependency with TileLang's C++ exports"
                )
            source = validate_source(source, revision)
            metadata = project_metadata(source / "pyproject.toml", uv)
            if metadata["project"]["version"] != version:
                raise ValueError(f"The selected checkout must provide the pinned version {version}")
            run(
                [
                    *uv,
                    "pip",
                    "install",
                    "--python",
                    arguments.python,
                    *metadata["build-system"]["requires"],
                    *metadata["project"]["dependencies"],
                ]
            )
            command = [
                *uv,
                "build",
                source,
                "--wheel",
                "--no-build-isolation",
                "--python",
                arguments.python,
                "--out-dir",
                temporary / "raw",
            ]
            if arguments.build_dir:
                command.extend(["-C", f"build-dir={arguments.build_dir.resolve()}"])
            command.extend(argument for name, value in definitions.items() for argument in ("-C", f"cmake.define.{name}={value}"))
            environment = {
                name: value for name, value in os.environ.items() if name not in {"CMAKE_ARGS", "PYTHONPATH", "TVM_TIRX_LIBRARY_PATH"}
            }
            run(command, env=environment)
            wheels = list((temporary / "raw").glob("*.whl"))
            if len(wheels) != 1:
                raise ValueError("The standalone dependency build must produce exactly one wheel")
            wheel = wheels[0]
            validate_wheel(wheel, version)
        if arguments.repair:
            from repair_tirx_wheel import repair

            wheel = repair(wheel, temporary / "repaired", "tirx", archs=arguments.archs)
        destination = output / wheel.name
        if wheel.resolve() != destination:
            shutil.copy2(wheel, destination)
    run([*uv, "pip", "install", "--python", arguments.python, "--force-reinstall", "--no-deps", destination])
    run([*uv, "pip", "install", "--python", arguments.python, f"apache-tvm-tirx=={version}", "--find-links", output])
    configuration = installed_configuration(arguments.python)
    if configuration["version"] != version:
        raise ValueError("The interpreter did not select the pinned standalone package")
    validate_backends(configuration, definitions)
    exported = {
        "TILELANG_USE_INSTALLED_TIRX": "ON",
        "tvm_tirx_DIR": configuration["cmake_dir"],
        "TVM_TIRX_PREPARED_WHEEL": str(destination),
        "UV_FIND_LINKS": str(output),
        "PIP_FIND_LINKS": str(output),
    }
    if arguments.github_env:
        with Path(arguments.github_env).open("a") as stream:
            for name, value in exported.items():
                if "\n" in value or "\r" in value:
                    raise ValueError("Build environment paths must not contain newlines")
                stream.write(f"{name}={value}\n")
    print(json.dumps(exported, indent=2), flush=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.environ.get("TVM_TIRX_SOURCE"))
    parser.add_argument("--wheel", default=os.environ.get("TVM_TIRX_WHEEL"))
    parser.add_argument("--repository", default=os.environ.get("TVM_TIRX_REPOSITORY"))
    parser.add_argument("--revision", default=os.environ.get("TVM_TIRX_REVISION"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "dist" / "tirx")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("-D", "--define", action="append", default=[])
    parser.add_argument("--github-env", nargs="?", const=os.environ.get("GITHUB_ENV"))
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--archs")
    arguments = parser.parse_args()
    if arguments.source and arguments.wheel:
        parser.error("Choose either a source checkout or a wheel")
    try:
        provision(arguments)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Standalone TIRx provisioning failed: {error}\n")


if __name__ == "__main__":
    main()
