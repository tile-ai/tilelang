import hashlib
import os
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[3]
SOURCE = Path(os.environ.get("TVM_TIRX_SOURCE") or PROJECT.parent / "tvm-tirx")
UPSTREAM_BASELINE = "9af5544df8553c35b6a85927cd2589f0a596f2fc"
UPSTREAM_PASS_HASHES = {
    "include/tvm/tirx/stmt_functor_with_analyzer.h": "c0035cf6051302062ec710e967ddf391f757e04bc9a4ebeac1f6ccc6d122028f",
    "src/tirx/ir/ir_mutator_with_analyzer.cc": "905c47f43fdb42866b639f1a4f19d090344f00cedd07c361167c29b9164fcb32",
    "src/tirx/ir/ir_visitor_with_analyzer.cc": "645abf2d635f50f63b5135f2818e9ab559735419cf7ad68498d303ce07813170",
    "src/tirx/transform/tvm_ffi_binder.cc": "861665e71cc183804954769a951e865f28ea433c6b56bb9381de1dc4b6c79054",
    "src/tirx/transform/tvm_ffi_binder.h": "95bb9c2d0b54407f8780594dd6c26effc36d23751303ad457065cbd4aef55cb8",
    "include/tvm/tirx/transform.h": "3b8e6ec42d3bc0e6285a726dae395b924c4f15b5505910c3ed93b964d529d01c",
    "include/tvm/tirx/op/attr.h": "539de169354e45ee82a0433e795e84e36ee3323647640e662aa2b3e7b1f138cb",
    "python/tvm_tirx/tirx/transform/transform.py": "5cc529a1ff343f139929420f81835ea94f00a4167db340a19cce1a8cd64379d3",
    "src/tirx/transform/storage_rewrite.cc": "aea86a85c5d763e8cf4dc89fcb79598ad07fe56c53c6b47edc55ab2ef5e5b3ae",
    "src/tirx/transform/split_host_device.cc": "fb883638e6868eb7141ecac435c83ebf0c3b18b8bb70fb73d90188304efea4c3",
    "src/tirx/transform/vectorize_loop.cc": "b3404cd96f1d92a262e0c8c48b75828d1ad5a4b7c3ec3d3cb30550991c6bd5ff",
    "src/tirx/transform/hoist_if.cc": "b870b16753d34bdc6a3d8eabda772147f44794873f6375d7f1df9fc755a99202",
    "src/tirx/transform/flatten_tensor.cc": "8574adf535a4d2c016082dee568099996cae76efdd3632de018f52b801039d43",
    "src/tirx/transform/flatten_tensor_utils.h": "6d2df865fe31fc72b28d4ab28af21c145bceb5b5b9b0bbac9bc756cf7d8b2d91",
    "src/tirx/transform/lower_tensor_layouts.cc": "2a1e54009afa7c387febdf78a5d6d9dfe8883e7671046b07e4e8d3829ee624d5",
    "src/tirx/transform/stmt_simplify.cc": "de0786d13a28bfd6e906e670b1ba5cf05f2ffcd113174d8dc01a2ac94f3c7c5b",
    "src/tirx/transform/remove_no_op.cc": "d5f0804fa7b78fb9a5283d37200c84ba6737b2eeb09661ad3d6fc9c0775d3c18",
    "src/tirx/transform/bind_target.cc": "4d267f599c19ad97037219fb76390f2ddeeedbfb5f3ef50ff46ee43662b7e71a",
    "src/tirx/transform/function_utils.cc": "c35d76a930f55310afb1c49f7b5caad647a4e41c42cbc0be795c2c5e7dc9edf1",
    "src/tirx/transform/force_narrow_index_to_i32.cc": "11dea35cabb19be3e7dcf049caec20883fa845428281a99d87ed0ffa2b3ebdba",
    "src/tirx/transform/narrow_datatype.cc": "138b02716f43d2a00469c278a75d5e3e097b7a9b44b9bcece5162ff0d9438699",
    "src/tirx/transform/unroll_loop.cc": "d9aa3ea4e7836db986b467db2749917a0e234f53705a39db03bcb1d2f857d955",
    "src/tirx/transform/lower_tvm_builtin.cc": "80aafb3f58d600ee621c1d55aa78dd61d2b02d4a7b22b0b955c082a7bedcc514",
    "src/tirx/transform/make_packed_api.cc": "86ebccb78bf8c04c75c32977e1c43fe46a24e3925675f700634d2a133c566a5b",
}


@pytest.mark.parametrize("relative,digest", UPSTREAM_PASS_HASHES.items())
def test_tirx_pass_sources_are_unmodified_upstream(relative, digest):
    source = SOURCE / relative
    if not source.is_file():
        pytest.skip("The selected upstream TIRx source is required for the ownership audit")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, (UPSTREAM_BASELINE, relative)


def test_tilelang_does_not_require_tirx_pass_extensions():
    cmake = (PROJECT / "cmake/load_tirx.cmake").read_text()
    for extension in ("FVectorizeCall", "OutlineDeviceRegions", "allow_local_inplace", "allow_nontrivial_else", "packed_api.h"):
        assert extension not in cmake


def test_downstream_passes_do_not_replace_tirx_global_registrations():
    sources = PROJECT / "src/transform/tirx"
    assert sources.is_dir()
    for source in sources.glob("*.cc"):
        text = source.read_text()
        assert '.def("tirx.transform.' not in text, source
        assert '#include "tvm_ffi_binder.h"' not in text, source
        assert "tvm-tirx/src/" not in text, source
    assert '"tl.FVectorizeCall"' in (PROJECT / "src/transform/tirx_passes.h").read_text()
