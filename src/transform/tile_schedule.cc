/*!
 * \file tile_schedule.cc
 * \brief Persistent tile schedule: 2D grid -> 1D wave loop for PersistentKernel.
 */

#include <optional>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "./common/attr.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::transform::PassContext;

namespace {

using tvm::ffi::Array;
using tvm::ffi::GetRef;
using tvm::ffi::Map;

constexpr const char *kPersistentKernel = "tilelang.is_persistent_kernel";

inline bool BlockRequestsPersistentTileSchedule(const Block &blk) {
  if (!blk->annotations.defined() || !blk->annotations.count(kPersistentKernel))
    return false;
  auto ann = blk->annotations.Get(kPersistentKernel);
  if (!ann)
    return false;
  // Kernel launch stores Python bool as Bool / IntImm in block annotations.
  if (const auto *imm = ann.value().as<IntImmNode>())
    return imm->value != 0;
  return true;
}

inline Stmt UnwrapHostRoot(Stmt s) {
  if (const auto *br = s.as<BlockRealizeNode>()) {
    if (IsHostMainBlock(br->block.get()))
      return br->block->body;
  }
  return s;
}

struct ThreadExtentAttr {
  IterVar iv;
  PrimExpr extent;
};

std::optional<Stmt> TryPersistent2DGrid(Stmt device_body, int sm_num) {
  // Expect: AttrStmt(blockIdx.x) -> AttrStmt(blockIdx.y) -> [thread attrs] ->
  // BlockRealize(tilelang_root)
  if (!device_body.as<AttrStmtNode>())
    return std::nullopt;

  ThreadExtentAttr blk_x{}, blk_y{};
  Stmt cursor = device_body;

  auto peel_block = [&](ThreadExtentAttr *out) -> bool {
    const auto *a = cursor.as<AttrStmtNode>();
    if (!a || a->attr_key != tir::attr::thread_extent)
      return false;
    auto iv = Downcast<IterVar>(a->node);
    if (iv->thread_tag != "blockIdx.x" && iv->thread_tag != "blockIdx.y" &&
        iv->thread_tag != "blockIdx.z")
      return false;
    out->iv = iv;
    out->extent = a->value;
    cursor = a->body;
    return true;
  };

  if (!peel_block(&blk_x))
    return std::nullopt;
  if (blk_x.iv->thread_tag != "blockIdx.x")
    return std::nullopt;
  if (!peel_block(&blk_y))
    return std::nullopt;
  if (blk_y.iv->thread_tag != "blockIdx.y")
    return std::nullopt;

  // No blockIdx.z before thread indices
  if (const auto *a = cursor.as<AttrStmtNode>()) {
    if (a->attr_key == tir::attr::thread_extent) {
      auto iv = Downcast<IterVar>(a->node);
      if (iv->thread_tag == "blockIdx.z")
        return std::nullopt;
    }
  }

  Array<AttrStmt> thread_wrappers;
  while (const auto *a = cursor.as<AttrStmtNode>()) {
    if (a->attr_key != tir::attr::thread_extent)
      break;
    auto iv = Downcast<IterVar>(a->node);
    const auto &tag = iv->thread_tag;
    if (!(tag == "threadIdx.x" || tag == "threadIdx.y" || tag == "threadIdx.z"))
      break;
    thread_wrappers.push_back(GetRef<AttrStmt>(a));
    cursor = a->body;
  }

  const auto *realize = cursor.as<BlockRealizeNode>();
  if (!realize)
    return std::nullopt;
  Block blk = realize->block;
  if (!IsDeviceMainBlock(blk.get()))
    return std::nullopt;
  if (!BlockRequestsPersistentTileSchedule(blk))
    return std::nullopt;

  PrimExpr Gx = blk_x.extent;
  PrimExpr Gy = blk_y.extent;
  Var bx_v = blk_x.iv->var;
  Var by_v = blk_y.iv->var;
  PrimExpr total_tiles = Gx * Gy;
  PrimExpr sm_imm = IntImm(DataType::Int(32), sm_num);
  PrimExpr waves = ceildiv(total_tiles, sm_imm);

  Var w = Var("w_tile_sched", waves.dtype());
  PrimExpr tile_linear = sm_imm * w + bx_v;

  Map<Var, PrimExpr> repl;
  repl.Set(bx_v, FloorMod(tile_linear, Gx));
  repl.Set(by_v, FloorDiv(tile_linear, Gx));

  Block new_blk = Downcast<Block>(Substitute(blk, repl));
  Stmt tile_body = new_blk->body;
  Stmt guarded = IfThenElse(LT(tile_linear, total_tiles), tile_body);
  ffi::Map<ffi::String, ffi::Any> wave_annotations;
  wave_annotations.Set(attr::kPipelinePhaseContinuation, Integer(1));
  Stmt wave_loop =
      For(w, 0, waves, ForKind::kSerial, guarded, ffi::Optional<IterVar>(),
          wave_annotations);

  auto bp = new_blk.CopyOnWrite();
  bp->body = wave_loop;
  if (bp->annotations.defined()) {
    bp->annotations.erase(kPersistentKernel);
  }

  Stmt inner =
      BlockRealize(realize->iter_values, realize->predicate, new_blk);
  for (int i = static_cast<int>(thread_wrappers.size()) - 1; i >= 0; --i) {
    const AttrStmtNode *tw = thread_wrappers[i].get();
    inner = AttrStmt(tw->node, tw->attr_key, tw->value, inner);
  }

  IterVar new_bx_iv = IterVar(Range(make_const(bx_v.dtype(), 0), sm_imm), bx_v,
                              blk_x.iv->iter_type, blk_x.iv->thread_tag);
  inner =
      AttrStmt(new_bx_iv, tir::attr::thread_extent, sm_imm, inner);
  return inner;
}

class TileSchedulePass {
public:
  static PrimFunc Run(PrimFunc f, int sm_num) {
    ICHECK_GT(sm_num, 0) << "TileSchedule num_persistent_blocks must be positive";

    Stmt body = f->body;
    Stmt host_peeled = UnwrapHostRoot(body);

    auto new_device = TryPersistent2DGrid(host_peeled, sm_num);
    if (!new_device.has_value())
      return f;

    Stmt new_body = new_device.value();
    if (host_peeled.get() != body.get()) {
      const auto *root_br = body.as<BlockRealizeNode>();
      ICHECK(root_br && IsHostMainBlock(root_br->block.get()));
      Block nb = root_br->block;
      nb.CopyOnWrite()->body = new_body;
      new_body = BlockRealize(root_br->iter_values, root_br->predicate, nb);
    }

    f.CopyOnWrite()->body = new_body;
    return f;
  }
};

} // namespace

tvm::transform::Pass TileSchedule(int num_persistent_blocks) {
  auto pass_func = [num_persistent_blocks](PrimFunc fn, IRModule m,
                                           PassContext ctx) {
    (void)m;
    (void)ctx;
    return TileSchedulePass::Run(std::move(fn), num_persistent_blocks);
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0, "tl.TileSchedule",
                                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.TileSchedule", TileSchedule);
}

} // namespace tl
} // namespace tvm
