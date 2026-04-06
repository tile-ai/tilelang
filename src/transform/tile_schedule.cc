/*!
 * \file tile_schedule.cc
 * \brief Persistent tile schedule: 2D grid -> 1D wave loop for PersistentKernel.
 */

#include <optional>
#include <unordered_set>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "./common/attr.h"
#include "./common/tile_scheduler.h"

namespace tvm {
namespace tl {

using namespace tir;
using tvm::transform::PassContext;

namespace {

using tvm::ffi::Array;
using tvm::ffi::GetRef;
using tvm::ffi::Map;
using namespace tile_scheduler;

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

bool ContainsWarpSpecialize(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (const auto *attr = node.as<AttrStmtNode>()) {
      if (attr->attr_key == "warp_specialize") {
        found = true;
      }
    }
  });
  return found;
}

bool UsesTileCoords(const Stmt &stmt, const Var &bx, const Var &by) {
  bool uses = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (const auto *var = node.as<VarNode>()) {
      if (var == bx.get() || var == by.get()) {
        uses = true;
      }
    }
  });
  return uses;
}

bool UsesOnlyAllowedVars(const PrimExpr &expr,
                        const std::unordered_set<const VarNode *> &allowed) {
  bool ok = true;
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (!ok) {
      return;
    }
    if (const auto *var = node.as<VarNode>()) {
      if (!allowed.count(var)) {
        ok = false;
      }
    }
  });
  return ok;
}

bool IsThreadRoleSpecializedStmt(
    const Stmt &stmt,
    const std::unordered_set<const VarNode *> &allowed_thread_vars) {
  if (ContainsWarpSpecialize(stmt)) {
    return true;
  }

  std::function<bool(const Stmt &)> visit = [&](const Stmt &s) -> bool {
    if (const auto *if_node = s.as<IfThenElseNode>()) {
      if (!UsesOnlyAllowedVars(if_node->condition, allowed_thread_vars)) {
        return false;
      }
      return true;
    }
    if (const auto *attr = s.as<AttrStmtNode>()) {
      return visit(attr->body);
    }
    if (const auto *let = s.as<LetStmtNode>()) {
      if (!UsesOnlyAllowedVars(let->value, allowed_thread_vars)) {
        return false;
      }
      return visit(let->body);
    }
    if (const auto *realize = s.as<BlockRealizeNode>()) {
      return visit(realize->block->body);
    }
    if (const auto *block = s.as<BlockNode>()) {
      return visit(block->body);
    }
    return false;
  };
  return visit(stmt);
}

Stmt WrapStmtWithPersistentLoop(const Stmt &stmt, const Map<Var, PrimExpr> &repl,
                                const TileWorkBinding &binding, const Var &w,
                                const PrimExpr &waves) {
  Stmt scheduled = Substitute(stmt, repl);
  Stmt guarded = IfThenElse(binding.is_valid, scheduled);
  ffi::Map<ffi::String, ffi::Any> wave_annotations;
  wave_annotations.Set(attr::kPipelinePhaseContinuation, Integer(1));
  return For(w, 0, waves, ForKind::kSerial, guarded, ffi::Optional<IterVar>(),
             wave_annotations);
}

std::optional<Stmt> TryRoleLocalPersistentBody(const Stmt &body,
                                               const Map<Var, PrimExpr> &repl,
                                               const TileWorkBinding &binding,
                                               const Var &w,
                                               const PrimExpr &waves,
                                               const Var &bx_v,
                                               const Var &by_v,
                                               const std::unordered_set<const VarNode *>
                                                   &allowed_thread_vars) {
  std::function<std::optional<Stmt>(const Stmt &)> visit =
      [&](const Stmt &stmt) -> std::optional<Stmt> {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      int region_start = -1;
      bool has_role_local = false;
      for (int i = 0; i < static_cast<int>(seq->seq.size()); ++i) {
        const Stmt &child = seq->seq[i];
        bool role_local =
            IsThreadRoleSpecializedStmt(child, allowed_thread_vars);
        bool uses_tiles = UsesTileCoords(child, bx_v, by_v);
        if (region_start < 0 && (role_local || uses_tiles)) {
          region_start = i;
        }
        has_role_local = has_role_local || role_local;
      }

      if (region_start < 0 || !has_role_local) {
        return std::nullopt;
      }

      Array<Stmt> rewritten;
      rewritten.reserve(seq->seq.size());
      for (int i = 0; i < region_start; ++i) {
        rewritten.push_back(seq->seq[i]);
      }

      Array<Stmt> uniform_suffix;
      auto flush_uniform_suffix = [&]() {
        if (uniform_suffix.empty()) {
          return;
        }
        Stmt uniform_stmt =
            uniform_suffix.size() == 1 ? uniform_suffix[0] : SeqStmt(uniform_suffix);
        rewritten.push_back(
            WrapStmtWithPersistentLoop(uniform_stmt, repl, binding, w, waves));
        uniform_suffix.clear();
      };

      for (int i = region_start; i < static_cast<int>(seq->seq.size()); ++i) {
        const Stmt &child = seq->seq[i];
        if (IsThreadRoleSpecializedStmt(child, allowed_thread_vars)) {
          flush_uniform_suffix();
          rewritten.push_back(
              WrapStmtWithPersistentLoop(child, repl, binding, w, waves));
        } else {
          uniform_suffix.push_back(child);
        }
      }
      flush_uniform_suffix();
      return SeqStmt(rewritten);
    }

    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      auto inner = visit(attr->body);
      if (!inner.has_value()) {
        return std::nullopt;
      }
      return AttrStmt(attr->node, attr->attr_key, attr->value, inner.value());
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      auto inner = visit(let->body);
      if (!inner.has_value()) {
        return std::nullopt;
      }
      return LetStmt(let->var, let->value, inner.value());
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      auto inner = visit(realize->block->body);
      if (!inner.has_value()) {
        return std::nullopt;
      }
      Block new_block = realize->block;
      new_block.CopyOnWrite()->body = inner.value();
      return BlockRealize(realize->iter_values, realize->predicate, new_block);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      auto inner = visit(block->body);
      if (!inner.has_value()) {
        return std::nullopt;
      }
      Block new_block = GetRef<Block>(block);
      new_block.CopyOnWrite()->body = inner.value();
      return new_block;
    }
    return std::nullopt;
  };

  return visit(body);
}

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
  std::unordered_set<const VarNode *> allowed_thread_vars;
  while (const auto *a = cursor.as<AttrStmtNode>()) {
    if (a->attr_key != tir::attr::thread_extent)
      break;
    auto iv = Downcast<IterVar>(a->node);
    const auto &tag = iv->thread_tag;
    if (!(tag == "threadIdx.x" || tag == "threadIdx.y" || tag == "threadIdx.z"))
      break;
    thread_wrappers.push_back(GetRef<AttrStmt>(a));
    allowed_thread_vars.insert(iv->var.get());
    cursor = a->body;
  }

  const auto *realize = cursor.as<BlockRealizeNode>();
  if (!realize)
    return std::nullopt;
  Block blk = realize->block;
  if (!IsDeviceMainBlock(blk.get()))
    return std::nullopt;

  PrimExpr Gx = blk_x.extent;
  PrimExpr Gy = blk_y.extent;
  Var bx_v = blk_x.iv->var;
  Var by_v = blk_y.iv->var;
  auto spec = ParseTileScheduleSpec(blk, sm_num);
  if (!spec.has_value()) {
    return std::nullopt;
  }
  ICHECK(spec->distribution_kind == DistributionKind::kPersistentStatic)
      << "TileSchedule currently only materializes persistent_static";

  PrimExpr total_tiles = Gx * Gy;
  PrimExpr worker_count = IntImm(DataType::Int(32), spec->num_persistent_workers);
  PrimExpr waves = ceildiv(total_tiles, worker_count);
  Var w = Var("w_tile_sched", waves.dtype());

  auto stripped_body_and_legacy = StripLegacyThreadblockSwizzleAttr::Run(blk->body);
  bool emit_override_warning = false;
  TileScheduleSpec resolved_spec =
      ResolveLegacyPermutation(spec.value(), stripped_body_and_legacy.second,
                               &emit_override_warning);
  if (emit_override_warning) {
    LOG(WARNING) << "PersistentKernel swizzle overrides T.use_swizzle() for "
                    "persistent kernels; TileSchedule will use the "
                    "PersistentKernel order/panel_size.";
  }

  TileWorkBinding binding =
      MakePersistentStaticTileWorkBinding(resolved_spec, bx_v, w, Gx, Gy);

  Map<Var, PrimExpr> repl;
  repl.Set(bx_v, binding.tile_x);
  repl.Set(by_v, binding.tile_y);
  Block stripped_blk = blk;
  stripped_blk.CopyOnWrite()->body = stripped_body_and_legacy.first;
  Block new_blk = Downcast<Block>(Substitute(stripped_blk, repl));
  auto role_local_body =
      TryRoleLocalPersistentBody(stripped_body_and_legacy.first, repl, binding,
                                 w, waves, bx_v, by_v, allowed_thread_vars);
  Stmt scheduled_body;
  if (role_local_body.has_value()) {
    scheduled_body = role_local_body.value();
  } else {
    Stmt tile_body = new_blk->body;
    Stmt guarded = IfThenElse(binding.is_valid, tile_body);
    ffi::Map<ffi::String, ffi::Any> wave_annotations;
    wave_annotations.Set(attr::kPipelinePhaseContinuation, Integer(1));
    scheduled_body = For(w, 0, waves, ForKind::kSerial, guarded,
                         ffi::Optional<IterVar>(), wave_annotations);
  }

  auto bp = new_blk.CopyOnWrite();
  bp->body = scheduled_body;
  if (bp->annotations.defined()) {
    bp->annotations.erase(kPersistentKernel);
    bp->annotations.erase(kTileScheduleKind);
    bp->annotations.erase(kTilePermutationKind);
    bp->annotations.erase(kTilePermutationPanelSize);
    bp->annotations.erase(kLegacyPersistentSwizzleOrder);
    bp->annotations.erase(kLegacyPersistentSwizzlePanelSize);
  }

  Stmt inner =
      BlockRealize(realize->iter_values, realize->predicate, new_blk);
  for (int i = static_cast<int>(thread_wrappers.size()) - 1; i >= 0; --i) {
    const AttrStmtNode *tw = thread_wrappers[i].get();
    inner = AttrStmt(tw->node, tw->attr_key, tw->value, inner);
  }

  IterVar new_bx_iv = IterVar(Range(make_const(bx_v.dtype(), 0), worker_count), bx_v,
                              blk_x.iv->iter_type, blk_x.iv->thread_tag);
  inner =
      AttrStmt(new_bx_iv, tir::attr::thread_extent, worker_count, inner);
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
