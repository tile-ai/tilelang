"""CANN selection and cache identity, runnable without TVM or an NPU."""

import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[3] / "tilelang/contrib/cann.py"
SPEC = importlib.util.spec_from_file_location("cann_context_under_test", SOURCE)
cann = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cann)


class CannContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sdk = self.root / "cann-directory-name-is-not-a-version"
        self.templates = self.root / "src"
        self.compiler = self.sdk / "compiler/bin/bisheng"
        files = {
            self.compiler: "compiler build",
            self.compiler.with_name("ld.lld"): "linker build",
            self.sdk / "asc/include/c_api/asc_simd.h": "// SIMD declarations\n",
            self.sdk / "asc/impl/instruction.h": "// SIMD implementation\n",
            self.sdk / "include/version/asc_devkit_version.h": '#define ASC_DEVKIT_VERSION_STR "1.0.0"\n'
            "#define ASC_DEVKIT_MAJOR 9\n#define ASC_DEVKIT_MINOR 1\n#define ASC_DEVKIT_PATCH 0\n",
            self.templates / "tl_templates/ascend/cann_compat.h": "// versioned adapter\n",
        }
        for component in ("asc-devkit", "bisheng-compiler", "runtime"):
            files[self.sdk / "share/info" / component / "version.info"] = "Version=9.1.0\n"
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        env = patch.dict(os.environ, {"ASCEND_HOME_PATH": str(self.sdk)}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def context(self):
        return cann.compile_context(self.compiler, self.templates)

    def test_numeric_version_not_directory_or_display_string(self):
        self.assertEqual(self.context()["numeric_version"], (9, 1, 0))

    def test_identical_context_reuses_identity(self):
        self.assertEqual(self.context()["fingerprint"], self.context()["fingerprint"])

    def test_same_version_header_edit_invalidates_identity(self):
        old = self.context()["fingerprint"]
        (self.sdk / "asc/impl/instruction.h").write_text("// revised SIMD implementation\n")
        self.assertNotEqual(old, self.context()["fingerprint"])

    def test_template_only_edit_invalidates_identity(self):
        old = self.context()["fingerprint"]
        (self.templates / "tl_templates/ascend/cann_compat.h").write_text("// different adapter\n")
        self.assertNotEqual(old, self.context()["fingerprint"])

    def check_same_stat_edit(self, target):
        # Model a same-length edit inside one filesystem timestamp tick.
        # Preserve the assertion; waiting/sleeping would hide the stale cache.
        frozen = target.stat()
        real_stat = Path.stat

        def stat(path, *args, **kwargs):
            return frozen if path == target else real_stat(path, *args, **kwargs)

        with patch.object(Path, "stat", stat):
            old = self.context()["fingerprint"]
            raw = target.read_bytes()
            target.write_bytes(raw.replace(b"//", b"/*", 1))
            self.assertEqual(len(raw), len(target.read_bytes()))
            self.assertNotEqual(old, self.context()["fingerprint"])

    def test_same_stat_template_edit_invalidates_identity(self):
        self.check_same_stat_edit(self.templates / "tl_templates/ascend/cann_compat.h")

    def test_same_stat_sdk_header_edit_invalidates_identity(self):
        self.check_same_stat_edit(self.sdk / "asc/impl/instruction.h")

    def test_toolchain_content_changes_invalidate_identity(self):
        for path in (self.compiler, self.compiler.with_name("ld.lld")):
            old = self.context()["fingerprint"]
            path.write_text(path.read_text() + " changed")
            self.assertNotEqual(old, self.context()["fingerprint"])

    def test_weekly_metadata_changes_invalidate_identity(self):
        old = self.context()["fingerprint"]
        for p in (self.sdk / "share/info").glob("*/version.info"):
            p.write_text("Version=9.1.0~weekly.new\n")
        self.assertNotEqual(old, self.context()["fingerprint"])

    def test_relocated_identical_contents_have_identical_identity(self):
        old = self.context()["fingerprint"]
        moved = self.root / "relocated"
        shutil.copytree(self.sdk, moved)
        with patch.dict(os.environ, {"ASCEND_HOME_PATH": str(moved)}):
            actual = cann.compile_context(moved / "compiler/bin/bisheng", self.templates)
        self.assertEqual(old, actual["fingerprint"])

    def test_rejects_mixed_components(self):
        (self.sdk / "share/info/runtime/version.info").write_text("Version=9.2.0\n")
        with self.assertRaisesRegex(RuntimeError, "Mixed CANN"):
            self.context()

    def test_rejects_header_metadata_disagreement(self):
        path = self.sdk / "include/version/asc_devkit_version.h"
        path.write_text(path.read_text().replace("ASC_DEVKIT_MINOR 1", "ASC_DEVKIT_MINOR 3"))
        with self.assertRaisesRegex(RuntimeError, "disagrees"):
            self.context()

    def test_rejects_missing_component_record(self):
        (self.sdk / "share/info/runtime/version.info").unlink()
        with self.assertRaisesRegex(RuntimeError, "metadata is missing"):
            self.context()

    def test_rejects_compiler_from_another_sdk(self):
        other = self.root / "other/bisheng"
        other.parent.mkdir()
        other.write_text("other compiler")
        with self.assertRaisesRegex(RuntimeError, "outside"):
            cann.compile_context(other, self.templates)

    def test_rejects_conflicting_environment(self):
        with (
            patch.dict(os.environ, {"ASCEND_TOOLKIT_HOME": str(self.root / "other")}),
            self.assertRaisesRegex(RuntimeError, "different SDKs"),
        ):
            self.context()

    def test_compiler_location_can_identify_sdk_without_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.context()["numeric_version"], (9, 1, 0))

    def test_architecture_subdirectory_environment_is_consistent(self):
        with patch.dict(os.environ, {"ASCEND_TOOLKIT_HOME": str(self.sdk / "x86_64-linux")}):
            self.assertEqual(self.context()["sdk_root"], str(self.sdk))


if __name__ == "__main__":
    unittest.main()
