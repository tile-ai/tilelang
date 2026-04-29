/*!
 * \file lower hopper intrin.cc
 * \brief Lower Hopper intrinsics cuda GPU(sm90+)
 *
 * Processes create_tma_descriptor / create_tma_im2col_descriptor calls:
 *   - Replaces each call with a typed __grid_constant__ descriptor Var.
 *   - Emits prefetch_tma_descriptor in the thread-leader prologue.
 *
 * Also handles tma_desc_slot calls:
 *   - Converts the raw workspace_ptr Var into a typed CUtensorMap* "global"
 *     scope Var so codegen emits it as a CUtensorMap* kernel parameter.
 *   - Leaves the tma_desc_slot Call in place (codegen emits ptr[idx]).
 *
 * Grid_constant descriptors are never replaced in-place because they are
 * read-only.  Mutable descriptor workspace is allocated on the host side
 * (CUtensorMap * kernel param) and written by tensormap_cp_fence_release.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <set>
#include <string>
#include <unordered_map>

#include "../op/builtin.h"
#include "../runtime/runtime.h"

namespace tvm {
namespace tl {

using namespace tir;

#if (CUDA_MAJOR_VERSION >= 12)

namespace {

/*!
 * \brief Scan for tma_desc_slot calls and collect workspace pointer Var names.
 */
class WorkSpacePtrCollector : public StmtExprVisitor {
public:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tma_desc_slot()) && !op->args.empty()) {
      if (auto var = op->args[0].as<VarNode>()) {
        workspace_ptr_names_.insert(var->name_hint);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  const std::set<std::string> &workspace_ptr_names() const {
    return workspace_ptr_names_;
  }

private:
  std::set<std::string> workspace_ptr_names_;
};

} // namespace

class LowerHopperIntrin : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f, bool disable_shuffle_elect) {
    PrimFuncNode *fptr = f.CopyOnWrite();

    // Scan for tma_desc_slot workspace pointer names.
    WorkSpacePtrCollector ws_collector;
    ws_collector(f->body);
    std::set<std::string> ws_ptr_names = ws_collector.workspace_ptr_names();

    LowerHopperIntrin substituter(disable_shuffle_elect, ws_ptr_names);
    fptr->body = substituter.VisitStmt(f->body);

    Map<Var, Array<PrimExpr>> init_desc_arg_map;
    Array<Stmt> prologue_stmts;
    Array<Stmt> epilogue_stmts;

    for (const auto &[call, var] : substituter.desc_map_) {
      Call alloc_desc = Call(DataType::Handle(), builtin::tvm_stack_alloca(),
                             {StringImm("tvm_ffi_any"), 16});
      Array<PrimExpr> init_desc_args;
      if (call->op.same_as(create_tma_descriptor())) {
        init_desc_args.push_back(StringImm(tvm_tensormap_create_tiled));
      } else if (call->op.same_as(create_tma_im2col_descriptor())) {
        init_desc_args.push_back(StringImm(tvm_tensormap_create_im2col));
      } else {
        CHECK(0) << call->op;
      }
      init_desc_args.push_back(var);
      init_desc_args.insert(init_desc_args.end(), call->args.begin(),
                            call->args.end());
      Call init_desc =
          Call(DataType::Handle(), builtin::tvm_call_packed(), init_desc_args);
      prologue_stmts.push_back(LetStmt(var, alloc_desc, Evaluate(init_desc)));
      init_desc_arg_map.Set(var, init_desc_args);
    }
    f = WithAttr(std::move(f), "tma_descriptor_args", init_desc_arg_map);

    // L2 persistent cache handling (unchanged).
    if (f->attrs.defined() && f->attrs->dict.count("l2_persistent_map")) {
      auto l2_map =
          f->GetAttr<Map<String, Array<PrimExpr>>>("l2_persistent_map");
      if (l2_map.defined()) {
        std::unordered_map<std::string, Buffer> name2buf;
        for (const auto &kv : f->buffer_map) {
          name2buf.emplace(kv.second->name, kv.second);
        }
        for (const auto &kv : l2_map.value()) {
          const std::string buf_name = kv.first;
          const Array<PrimExpr> &args = kv.second;
          if (name2buf.count(buf_name) == 0) continue;
          const Buffer &buf = name2buf.at(buf_name);
          PrimExpr base_ptr = buf->data;
          if (buf->elem_offset.defined() && !is_zero(buf->elem_offset)) {
            PrimExpr byte_offset =
                buf->elem_offset *
                IntImm(buf->elem_offset.dtype(), buf->dtype.bytes());
            base_ptr =
                Call(DataType::Handle(), builtin::handle_add_byte_offset(),
                     {base_ptr, byte_offset});
          }
          Array<PrimExpr> packed_args;
          packed_args.push_back(
              StringImm(tvm_cuda_stream_set_access_policy_window));
          packed_args.push_back(base_ptr);
          ICHECK_GE(args.size(), 2);
          packed_args.push_back(args[1]);
          packed_args.push_back(args[0]);
          prologue_stmts.push_back(Evaluate(Call(
              DataType::Int(32), builtin::tvm_call_packed(), packed_args)));
        }
        Array<PrimExpr> reset_args;
        reset_args.push_back(
            StringImm(tvm_cuda_stream_reset_access_policy_window));
        epilogue_stmts.push_back(Evaluate(
            Call(DataType::Int(32), builtin::tvm_call_packed(), reset_args)));
      }
    }

    // Add typed workspace vars to f->params and buffer_map so MakePackedAPI
    // includes them as API arguments with the correct handle type.
    if (!substituter.ws_ptr_var_map_.empty()) {
      for (const auto &kv : substituter.ws_ptr_var_map_) {
        Var ws_ptr = kv.second;
        // Add to function params
        fptr->params.push_back(ws_ptr);
        // One CUtensorMap workspace slot (128 bytes).
        Buffer ws_buf = decl_buffer({IntImm(DataType::Int(32), 128)},
                                    DataType::UInt(8), ws_ptr->name_hint,
                                    "global");
        ws_buf.CopyOnWrite()->data = ws_ptr;
        fptr->buffer_map.Set(ws_ptr, ws_buf);
      }
    }

    // Stitch prologue statements before the original body.
    if (!prologue_stmts.empty()) {
      Stmt seq = prologue_stmts.size() == 1 ? prologue_stmts[0]
                                            : SeqStmt(prologue_stmts);
      fptr->body = SeqStmt({seq, fptr->body});
    }
    if (!epilogue_stmts.empty()) {
      Stmt seq_end = epilogue_stmts.size() == 1 ? epilogue_stmts[0]
                                                : SeqStmt(epilogue_stmts);
      fptr->body = SeqStmt({fptr->body, seq_end});
    }
    return f;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        auto body = StmtExprMutator::VisitStmt(op->body);
        if (prefetch_calls_.empty()) {
          return AttrStmt(op->node, op->attr_key, op->value, body);
        } else {
          Array<Stmt> stmt_seq;
          PrimExpr condition;
          if (!disable_shuffle_elect_) {
            condition = Call(DataType::Bool(), tl_shuffle_elect(), {0});
          } else {
            condition = EQ(iv->var, 0);
          }
          auto stmts = prefetch_calls_;
          auto stmt_ = IfThenElse(condition,
                                  stmts.size() > 1 ? SeqStmt(stmts) : stmts[0]);
          stmt_seq.push_back(stmt_);
          stmt_seq.push_back(body);
          Stmt result = SeqStmt(stmt_seq);
          prefetch_calls_.clear();
          return AttrStmt(op->node, op->attr_key, op->value, result);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(create_tma_descriptor()) ||
        call->op.same_as(create_tma_im2col_descriptor())) {
      // All descriptors are grid_constant (read-only) with prefetch.
      Var var;
      auto iter = desc_map_.find(tvm::ffi::GetRef<Call>(call));
      if (iter != desc_map_.end()) {
        var = iter->second;
      } else {
        String name = call->args[2].as<Var>().value()->name_hint;
        var = Var(name + "_desc",
                  PointerType(PrimType(cuTensorMapType()), "grid_constant"));
        desc_map_[tvm::ffi::GetRef<Call>(call)] = var;
        prefetch_calls_.push_back(
            Evaluate(Call(DataType::Handle(), builtin::call_extern(),
                          {StringImm("tl::prefetch_tma_descriptor"), var})));
      }
      return var;
    } else if (call->op.same_as(tma_desc_slot())) {
      // Convert raw workspace_ptr Var to typed CUtensorMap* parameter.
      // The first arg is workspace_ptr, second is slot index.
      auto args = call->args;
      if (auto var = args[0].as<VarNode>()) {
        if (ws_ptr_names_.count(var->name_hint) > 0) {
          auto it = ws_ptr_var_map_.find(var->name_hint);
          if (it != ws_ptr_var_map_.end()) {
            args.Set(0, it->second);
          } else {
            Var typed = Var(var->name_hint,
                            PointerType(PrimType(cuTensorMapType()), "global"));
            ws_ptr_var_map_[var->name_hint] = typed;
            args.Set(0, typed);
          }
        }
      }
      if (args.same_as(call->args)) {
        return tvm::ffi::GetRef<PrimExpr>(call);
      }
      return Call(call->dtype, call->op, args);
    } else {
      return StmtExprMutator::VisitExpr_(call);
    }
  }

private:
  Array<Stmt> prefetch_calls_;
  std::unordered_map<Call, Var, StructuralHash, ExprDeepEqual> desc_map_;
  std::set<std::string> ws_ptr_names_;
  std::unordered_map<std::string, Var> ws_ptr_var_map_;
  LowerHopperIntrin(bool disable_shuffle_elect,
                    const std::set<std::string> &ws_ptr_names)
      : disable_shuffle_elect_(disable_shuffle_elect),
        ws_ptr_names_(ws_ptr_names) {}
  bool disable_shuffle_elect_;
};

using namespace tir::transform;

tvm::transform::Pass LowerHopperIntrin() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool disable_shuffle_elect =
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value();
    return LowerHopperIntrin::Substitute(f, disable_shuffle_elect);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerHopperIntrin", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerHopperIntrin", LowerHopperIntrin);
}
#endif // (CUDA_MAJOR_VERSION >= 12)

} // namespace tl
} // namespace tvm
