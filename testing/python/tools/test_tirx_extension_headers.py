import os
from pathlib import Path
import re

import pytest


PROJECT = Path(__file__).resolve().parents[3]
HEADERS = PROJECT / "src" / "tirx_compat"
SOURCE = Path(os.environ.get("TVM_TIRX_SOURCE") or PROJECT.parent / "tvm-tirx")
FORMER_PUBLIC_HEADERS = {
    "c.h": "codegen/c.h",
    "source_base.h": "codegen/source_base.h",
    "build_common.h": "codegen/build_common.h",
    "launch_param.h": "runtime/launch_param.h",
    "metadata.h": "runtime/metadata.h",
}


def native_header(header, native):
    source = SOURCE / native
    public_name = FORMER_PUBLIC_HEADERS[header]
    public = SOURCE / "include" / "tvm" / public_name
    if source.is_file():
        text = source.read_text()
        if f"#include <tvm/{public_name}>" not in text:
            return text
    if public.is_file():
        return public.read_text()
    pytest.skip("The matching TIRx source is needed to compare its private declarations")


@pytest.mark.parametrize(
    "header,native",
    [
        ("build_common.h", "src/codegen/build_common.h"),
        ("launch_param.h", "src/runtime/launch_param.h"),
        ("metadata.h", "src/runtime/metadata.h"),
    ],
)
def test_private_metadata_helpers_match_the_dependency(header, native):
    def normalize(text):
        text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.DOTALL)
        text = re.sub(r"^#include[^\n]*", "", text, flags=re.MULTILINE)
        while True:
            joined = re.sub(
                r'("(?:\\.|[^"\\])*")\s+("(?:\\.|[^"\\])*")',
                lambda match: match.group(1)[:-1] + match.group(2)[1:],
                text,
            )
            if joined == text:
                break
            text = joined
        return re.sub(r"\s+", "", text)

    assert normalize((HEADERS / header).read_text()) == normalize(native_header(header, native))


def test_private_extension_headers_do_not_include_uninstalled_tirx_headers():
    removed_headers = (
        "tvm/codegen/c.h",
        "tvm/codegen/source_base.h",
        "tvm/codegen/build_common.h",
        "tvm/runtime/launch_param.h",
        "tvm/runtime/metadata.h",
        "tvm/tirx/data_type_rewriter.h",
    )
    for header in HEADERS.glob("*.h"):
        text = header.read_text()
        for removed in removed_headers:
            assert f"#include <{removed}>" not in text, header
