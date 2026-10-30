#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the License in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Probe the SoC identity of the current environment.

Get the chain (npu-smi's Chip Name as short-soc-version is not trusted and is prohibited from being used for model identification - issue #587):
    1. full-soc-version (full model, such as Ascend950PR_9579):
       Prefer Chip Info for asys `info -r=hardware`; fallback to DSMI (dsmi_get_chip_info) on failure.
    2. NpuArch (such as 3510): Prefer asys' Arch Info; fall back to ini file on failure (full-soc-version exact match).
    3. short-soc-version (such as Ascend950)/CCE_AIV_version/variant_dir: from ini file.
       Among them, short-soc-version is used as the first parameter of AddConfig() of the operator prototype definition file xxx operator_def.cpp
       (such as this->AICore().AddConfig("ascend950", aicConfig)).

Usage:
    python3 get_npu_arch.py # Human reading report (including evidence chain comments)
    python3 get_npu_arch.py --raw # Only output raw NpuArch values, such as 3510
    python3 get_npu_arch.py --json # Machine-readable JSON

Prerequisite: source CANN installation directory set_env.sh (asys and libdrvdsmi_host.so depend on its PATH / LD_LIBRARY_PATH).
"""

import ctypes
import json
import logging
import os
import platform
import re
import subprocess
import sys

_LOGGER = logging.getLogger(__name__)

MAX_CHIP_NAME = 32


class HalChipInfo(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_char * MAX_CHIP_NAME),
        ("name", ctypes.c_char * MAX_CHIP_NAME),
        ("version", ctypes.c_char * MAX_CHIP_NAME),
    ]


# ---------------------------------------------------------------------------
# CANN home positioning (follow the original logic)
# ---------------------------------------------------------------------------


def _derive_from_opp_path():
    opp = os.environ.get("ASCEND_OPP_PATH", "")
    if opp and opp.endswith("/opp"):
        toolkit = opp[:-4]
        if os.path.isdir(toolkit) and os.path.isdir(os.path.join(toolkit, "compiler")):
            return toolkit
    return None


def _resolve_toolkit_path(base_path):
    if os.path.isdir(os.path.join(base_path, "compiler")):
        return base_path

    toolkit_dir = os.path.join(base_path, "ascend-toolkit")
    if os.path.isdir(toolkit_dir):
        candidates = []
        for d in os.listdir(toolkit_dir):
            if d == "latest":
                continue
            dpath = os.path.join(toolkit_dir, d)
            if os.path.isdir(dpath) and os.path.isdir(os.path.join(dpath, "compiler")):
                candidates.append(d)
        candidates.sort(reverse=True)
        if candidates:
            return os.path.join(toolkit_dir, candidates[0])

        latest = os.path.join(toolkit_dir, "latest")
        if os.path.islink(latest):
            real = os.path.realpath(latest)
            if os.path.isdir(real) and os.path.isdir(os.path.join(real, "compiler")):
                return real

    cann_candidates = []
    for d in os.listdir(base_path):
        if not d.startswith("cann-"):
            continue
        dpath = os.path.join(base_path, d)
        if os.path.isdir(dpath) and os.path.isdir(os.path.join(dpath, "compiler")):
            cann_candidates.append(d)
    cann_candidates.sort(reverse=True)
    if cann_candidates:
        return os.path.join(base_path, cann_candidates[0])

    return None


def get_cann_home():
    derived = _derive_from_opp_path()
    for var in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME"):
        path = os.environ.get(var, "")
        if path and os.path.isdir(path):
            resolved = _resolve_toolkit_path(path)
            if resolved:
                return resolved

    if derived:
        return derived

    for var in ("ASCEND_HOME_PATH", "ASCEND_CANN_HOME"):
        path = os.environ.get(var, "")
        if path and os.path.isdir(path):
            resolved = _resolve_toolkit_path(path)
            if resolved:
                return resolved
    raise RuntimeError(
        "Cannot locate CANN toolkit installation. Set one of: ASCEND_TOOLKIT_HOME, ASCEND_HOME, ASCEND_HOME_PATH, ASCEND_CANN_HOME"
    )


def get_arch_dir():
    return f"{platform.machine()}-linux"


# ---------------------------------------------------------------------------
# Layer 1: full-soc-version detection
# ---------------------------------------------------------------------------


def _find_asys():
    """Locate the asys executable file: first under ASCEND_HOME_PATH/tools, followed by PATH."""
    if os.environ.get("ASCEND_HOME_PATH"):
        cand = os.path.join(os.environ["ASCEND_HOME_PATH"], "tools", "ascend_system_advisor", "asys", "asys")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    for cand in (
        os.path.join(os.path.expanduser("~"), "Ascend", "tools", "ascend_system_advisor", "asys", "asys"),
        "/usr/local/Ascend/tools/ascend_system_advisor/asys/asys",
    ):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    # Search in PATH (only the command name is returned, which will be parsed by subprocess)
    from shutil import which

    return which("asys")


_ASYS_HW_CACHE = {}


def _run_asys_hardware(asys_cmd):
    """Execute asys info -r=hardware, return the original stdout; return None on failure. Cache once in the process."""
    if asys_cmd in _ASYS_HW_CACHE:
        return _ASYS_HW_CACHE[asys_cmd]
    try:
        result = subprocess.run(
            [asys_cmd, "info", "-r=hardware"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        output = None
    _ASYS_HW_CACHE[asys_cmd] = output
    return output


def _parse_asys_field(output, label):
    """Extract single-valued fields from asys table output, returning (value, evidence_line). The first column matches label exactly."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if cells and cells[0] == label and len(cells) >= 2:
                return cells[1], stripped
    return None, None


def _parse_full_soc_from_chip_info(chip_info_value):
    """Chip Info value is in the form of 'Ascend 950PR_9579 V100', remove the version segment and get 'Ascend950PR_9579'.

    Version segment identification: The last purely alphanumeric token separated by spaces and starting with V is discarded directly.
    """
    tokens = chip_info_value.split()
    if len(tokens) < 2:
        return chip_info_value.replace(" ", "")
    if len(tokens) >= 3 and re.fullmatch(r"V\w+", tokens[-1]):
        tokens = tokens[:-1]
    return "".join(tokens)  # 'Ascend' + '950PR_9579'


def probe_full_soc_via_asys():
    """Preferred: asys info -r=hardware's Chip Info. Return full_soc or None.

    Note: Some asys versions do not have the Arch Info field, and the NpuArch layer detects it separately.
    """
    asys_cmd = _find_asys()
    if not asys_cmd:
        _LOGGER.debug("asys not found")
        return None
    output = _run_asys_hardware(asys_cmd)
    if not output:
        _LOGGER.debug("asys info -r=hardware returned nothing")
        return None
    chip_info, evidence = _parse_asys_field(output, "Chip Info")
    if not chip_info:
        return None
    full_soc = _parse_full_soc_from_chip_info(chip_info)
    if not full_soc or not full_soc.startswith(("Ascend", "Kirin")):
        return None
    return full_soc


def probe_full_soc_via_dsmi():
    """Alternative: DSMI dsmi_get_chip_info (TTK dsmi_interface mode, single device query). Returns full_soc or None."""
    try:
        dll = ctypes.CDLL("libdrvdsmi_host.so")
        device_count = (ctypes.c_int * 1)()
        dll.dsmi_get_device_count.restype = ctypes.c_int
        if dll.dsmi_get_device_count(device_count) != 0 or device_count[0] <= 0:
            _LOGGER.debug("dsmi_get_device_count failed or no device")
            return None

        device_id = ctypes.c_int(0)
        info = HalChipInfo()
        dll.dsmi_get_chip_info.restype = ctypes.c_int
        if dll.dsmi_get_chip_info(device_id, ctypes.byref(info)) != 0:
            _LOGGER.debug("dsmi_get_chip_info failed")
            return None
    except (OSError, AttributeError) as e:
        # OSError: Library cannot be loaded; AttributeError: Driver version mismatch causes dsmi_* symbols to be missing
        _LOGGER.debug("DSMI probe unavailable: %s", e)
        return None

    chip_type = info.type.decode().strip()
    chip_name = info.name.decode().strip()
    full_soc = chip_type + chip_name
    if not full_soc.startswith(("Ascend", "Kirin")):
        return None
    return full_soc


def probe_full_soc():
    """full-soc-version: asys Chip Info is the first choice, DSMI is the best choice. Return (full_soc, source) or None."""
    result = probe_full_soc_via_asys()
    if result:
        return result, "asys"
    result = probe_full_soc_via_dsmi()
    if result:
        return result, "dsmi"
    return None


# ---------------------------------------------------------------------------
# Layer 2: NpuArch detection
# ---------------------------------------------------------------------------


def probe_npu_arch_via_asys():
    """Preferred: Arch Info of asys info -r=hardware (some versions do not have this field).

    Return (npu_arch_str, evidence_line) or None.
    """
    asys_cmd = _find_asys()
    if not asys_cmd:
        return None
    output = _run_asys_hardware(asys_cmd)
    if not output:
        return None
    arch, evidence = _parse_asys_field(output, "Arch Info")
    if arch is None:
        return None
    arch = arch.strip()
    if not re.fullmatch(r"\d{4}", arch):
        return None
    return arch, evidence


# ---------------------------------------------------------------------------
# Layer 3: ini query (NpuArch bottom line + short-soc-version / CCE_AIV_version / variant_dir)
# ---------------------------------------------------------------------------


def _ini_has_soc_version(ini_path, full_soc):
    """Check whether a single ini contains the exact line SoC_version=full_soc (full file match line by line)."""
    try:
        with open(ini_path, errors="ignore") as f:
            for line in f:
                if line.strip() == f"SoC_version={full_soc}":
                    return True
    except OSError:
        pass
    return False


def find_ini_for_soc(cann_home, full_soc):
    """Exact match ini by SoC_version field (lookup_arch_variant mode, prefix ambiguity is prohibited)."""
    config_dir = os.path.join(cann_home, get_arch_dir(), "data", "platform_config")
    if not os.path.isdir(config_dir):
        return None
    for name in sorted(os.listdir(config_dir)):
        if not name.endswith(".ini"):
            continue
        ini_path = os.path.join(config_dir, name)
        if _ini_has_soc_version(ini_path, full_soc):
            return ini_path
    return None


def read_ini_fields(ini_path):
    """Read the [version] section key field."""
    fields = {}
    in_version_section = False
    with open(ini_path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "[version]":
                in_version_section = True
                continue
            if in_version_section and line.startswith("["):
                break
            if in_version_section and "=" in line:
                key, val = line.split("=", 1)
                fields[key.strip()] = val.strip()
    return fields


def variant_dir_from_aiv(ccec_aiv_version):
    """dav-c310-vec -> dav_c310; return the original value if it is not in this format."""
    m = re.match(r"^dav-([a-z0-9]+)-vec$", ccec_aiv_version or "")
    if m:
        return f"dav_{m.group(1)}"
    return ccec_aiv_version


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _apply_ini_fields(result, ini_path):
    """Write the ini field into result: short-soc-version / CCE_AIV_version / variant_dir; NpuArch knows the truth."""
    fields = read_ini_fields(ini_path)
    result["ini_path"] = ini_path
    result["short_soc"] = fields.get("Short_SoC_version")
    result["ccec_aiv_version"] = fields.get("CCEC_AIV_version")
    result["variant_dir"] = variant_dir_from_aiv(fields.get("CCEC_AIV_version", ""))
    ini_arch = fields.get("NpuArch")
    if not result["npu_arch"]:
        if ini_arch:
            result["npu_arch"], result["npu_arch_source"] = ini_arch, "ini"
    elif ini_arch and ini_arch != result["npu_arch"]:
        result["warnings"].append(f"NpuArch is inconsistent: asys={result['npu_arch']}, ini={ini_arch} (asys shall prevail, please check)")


def _probe_npu_count():
    """Device count (asys NPU Count), returns None on failure."""
    asys_cmd = _find_asys()
    if not asys_cmd:
        return None
    output = _run_asys_hardware(asys_cmd)
    if not output:
        return None
    count, _ = _parse_asys_field(output, "NPU Count")
    if count is None:
        return None
    digits = re.sub(r"\D", "", count)
    return int(digits) if digits else None


def probe_all():
    """Detect by level and return a dict (including each source and evidence)."""
    result = {
        "full_soc": None,
        "full_soc_source": None,
        "npu_arch": None,
        "npu_arch_source": None,
        "short_soc": None,
        "ccec_aiv_version": None,
        "variant_dir": None,
        "ini_path": None,
        "npu_count": None,
        "warnings": [],
    }

    # Layer 1: full-soc-version
    probed = probe_full_soc()
    if not probed:
        result["warnings"].append(
            "full-soc-version probe failed: neither asys nor DSMI available or no device."
            "Please make sure you have sourced set_env.sh in the CANN installation directory."
        )
        result["npu_count"] = _probe_npu_count()
        return result
    result["full_soc"], result["full_soc_source"] = probed

    # Layer 2: NpuArch (asys preferred)
    arch = probe_npu_arch_via_asys()
    if arch:
        result["npu_arch"], result["npu_arch_source"] = arch[0], "asys"

    # Layer 3: ini (short-soc-version / CCE_AIV_version / variant_dir; NpuArch cover)
    try:
        cann_home = get_cann_home()
    except RuntimeError as e:
        result["warnings"].append(f"{e}; short-soc-version/CCE_AIV_version/variant_dir cannot be obtained from ini")
        cann_home = None

    if cann_home:
        ini_path = find_ini_for_soc(cann_home, result["full_soc"])
        if ini_path:
            _apply_ini_fields(result, ini_path)
        else:
            result["warnings"].append(
                f"There is no ini with SoC_version={result['full_soc']} under platform_config,"
                "short-soc-version/CCE_AIV_version/variant_dir unknown"
            )

    if not result["npu_arch"]:
        result["warnings"].append("NpuArch detection failed: asys Arch Info and ini are not provided")

    result["npu_count"] = _probe_npu_count()
    return result


# ---------------------------------------------------------------------------
# Output (refer to lookup_arch_variant.sh for printing style: value + end-of-line purpose comment to enhance memory)
# ---------------------------------------------------------------------------


def _format_report(r):
    lines = []
    lines.append(f"full-soc-version={r['full_soc']} (source={r['full_soc_source']})")
    if r["npu_arch"] is not None:
        lines.append(
            f"NpuArch={r['npu_arch']} (source={r['npu_arch_source']})   "
            "# __NPU_ARCH__ macro branch / --npu-arch compiled value; only this macro branch is used to read the header file"
        )
    if r["short_soc"]:
        lines.append(
            f"short-soc-version={r['short_soc']}   "
            "# Operator prototype definition xxx operator_def.cpp The first parameter of AddConfig() (such as "
            'this->AICore().AddConfig("ascend950", aicConfig))'
        )
    if r["ccec_aiv_version"]:
        lines.append(f"CCE_AIV_version={r['ccec_aiv_version']}")
    if r["variant_dir"]:
        lines.append(
            f"variant_dir={r['variant_dir']}   "
            f"# When reading the source code in the CANN installation directory, only look at the files under **/{r['variant_dir']}/"
        )
    if r["ini_path"]:
        lines.append(f"ini={r['ini_path']}")
    if r["npu_count"] is not None:
        lines.append(f"npu_count={r['npu_count']}")
    lines.append(f"dav-{r['npu_arch']}" if r["npu_arch"] else "dav-UNKNOWN")
    for w in r["warnings"]:
        lines.append(f"WARNING: {w}")
    return "\n".join(lines)


def main() -> int:
    args = sys.argv[1:]
    raw_mode = "--raw" in args
    json_mode = "--json" in args

    r = probe_all()

    if raw_mode:
        if r["npu_arch"]:
            print(r["npu_arch"])
            return 0
        for w in r["warnings"]:
            print(f"WARNING: {w}", file=sys.stderr)
        return 1

    if json_mode:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if (r["full_soc"] and r["npu_arch"]) else 1

    if not r["full_soc"]:
        print(_format_report(r))
        return 1

    print(_format_report(r))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(main())
