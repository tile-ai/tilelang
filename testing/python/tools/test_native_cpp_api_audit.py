"""The API audit must distinguish native op registration from public API drift."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]


def _load_audit():
    spec = importlib.util.spec_from_file_location("tilelang_native_api_audit", ROOT / "maint/scripts/audit_cpp_api_style.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _registered(audit, paths):
    args = SimpleNamespace(paths=[str(path) for path in paths], include_templates=False, include_runtime_shims=False, include_vendor=False)
    return audit.collect_registered_op_names(args)


def test_real_native_metal_accessors_are_registered(monkeypatch):
    monkeypatch.chdir(ROOT)
    audit = _load_audit()
    source = ROOT / "src/metal/op/builtin.cc"
    names = _registered(audit, [source])
    expected = {"cooperative_tensor_fill", "cooperative_tensor_load", "cooperative_tensor_store", "cooperative_tensor_multiply_accumulate"}
    assert expected <= names
    findings = audit.audit_file(source, names)
    assert not [row for row in findings if row.rule == "TLCPP001" and row.symbol in expected | {"def", "TVM_FFI_STATIC_INIT_BLOCK"}]


def test_real_native_core_accessors_are_registered(monkeypatch):
    monkeypatch.chdir(ROOT)
    audit = _load_audit()
    source = ROOT / "src/op/builtin.h"
    names = _registered(audit, [source])
    assert {"access_ptr", "atomic_load_elem_op", "clamp"} <= names
    findings = audit.audit_file(source, names)
    assert not [row for row in findings if row.rule == "TLCPP001" and row.symbol in names]


def test_exports_keep_unregistered_function_and_field_diagnostics(tmp_path):
    audit = _load_audit()
    source = tmp_path / "api.h"
    source.write_text(
        "TVM_TIRX_DLL const Op &not_registered();\n"
        "class TVM_TIRX_DLL BadNode {\npublic:\n  int camelCase;\n};\n"
        "TVM_TIRX_DLL void suppressed_name(); // NOLINT\n"
        "TVM_RUNTIME_DLL void also_bad();\n"
    )
    findings = {(row.rule, row.symbol) for row in audit.audit_file(source, set())}
    assert ("TLCPP001", "not_registered") in findings
    assert ("TLCPP001", "also_bad") in findings
    assert ("TLCPP003", "camelCase") in findings
    assert ("TLCPP001", "suppressed_name") not in findings


def test_lookup_and_unrelated_string_do_not_exempt_functions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    audit = _load_audit()
    source = tmp_path / "ops.cc"
    source.write_text(
        'const char* unrelated = "tl.not_registered";\n'
        'const Op& lookup_only() { return Op::Get("tl.lookup_only"); }\n'
        'TVM_FFI_STATIC_INIT_BLOCK() {\n  OpDef named("tl.registered");\n}\n'
    )
    names = _registered(audit, [source])
    assert names == {"registered"}
    assert ("TLCPP001", "lookup_only") in {(row.rule, row.symbol) for row in audit.audit_file(source, names)}


def test_inline_initializer_keeps_next_public_declaration(tmp_path):
    audit = _load_audit()
    source = tmp_path / "api.h"
    source.write_text('TVM_FFI_STATIC_INIT_BLOCK() { OpDef named("tl.registered"); }\nvoid unregistered_name();\n')
    findings = {(row.rule, row.symbol) for row in audit.audit_file(source, {"registered"})}
    assert ("TLCPP001", "unregistered_name") in findings
    assert ("TLCPP001", "named") not in findings
