/*!
 * \file lower hopper intrin.cc
 * \brief Lower Hopper intrinsics cuda GPU(sm90+)
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../runtime/runtime.h"

namespace tvm {
namespace tl {

using namespace tir;

#if (CUDA_MAJOR_VERSION >= 12)
namespace {

Stmt MakeSeqOrSingle(const Array<Stmt> &stmts) {
  ICHECK(!stmts.empty());
  return stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
}

Array<Var> GetPrologueInitialDefs(const PrimFunc &f) {
  Array<Var> defs = f->params;
  for (const auto &[_, buffer] : f->buffer_map) {
    defs.push_back(buffer->data);
  }
  return defs;
}

// Host-side setup emitted by this pass can reference buffers produced by
// earlier allocation passes. Place it at the first statement where TVM's own
// undefined-var analysis agrees that all referenced variables are available.
class DependencyAwarePrologueInserter : public StmtExprMutator {
public:
  DependencyAwarePrologueInserter(Stmt prologue, const Array<Var> &initial_defs)
      : prologue_(std::move(prologue)),
        required_vars_(UndefinedVars(prologue_, initial_defs)) {}

  Stmt Insert(const Stmt &body) {
    if (required_vars_.empty()) {
      inserted_ = true;
      return SeqStmt({prologue_, body});
    }

    Stmt result = VisitStmt(body);
    ICHECK(inserted_)
        << "Unable to place Hopper prologue after its variable definitions; "
        << "remaining undefined vars: " << required_vars_;
    return result;
  }

private:
  bool Ready() const { return required_vars_.empty(); }

  void RemoveRequiredVar(const Var &var) {
    Array<Var> remaining;
    bool removed = false;
    for (const Var &required : required_vars_) {
      if (!removed && required.same_as(var)) {
        removed = true;
        continue;
      }
      remaining.push_back(required);
    }
    required_vars_ = remaining;
  }

  Stmt InsertBefore(const Stmt &stmt) {
    inserted_ = true;
    return SeqStmt({prologue_, stmt});
  }

  template <typename VisitScope>
  Stmt VisitAllocationScope(const Var &var, VisitScope visit_scope) {
    Array<Var> saved_required_vars = required_vars_;
    RemoveRequiredVar(var);
    Stmt result = visit_scope();
    if (!inserted_) {
      required_vars_ = saved_required_vars;
    }
    return result;
  }

  Stmt VisitStmt(const Stmt &stmt) final {
    if (inserted_) {
      return stmt;
    }
    if (Ready()) {
      return InsertBefore(stmt);
    }
    return StmtExprMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const AllocateNode *op) final {
    return VisitAllocationScope(
        op->buffer_var, [&]() { return StmtExprMutator::VisitStmt_(op); });
  }

  Stmt VisitStmt_(const AllocateConstNode *op) final {
    return VisitAllocationScope(
        op->buffer_var, [&]() { return StmtExprMutator::VisitStmt_(op); });
  }

  Stmt prologue_;
  Array<Var> required_vars_;
  bool inserted_ = false;
};

Stmt InsertAfterRequiredDefinitions(const Stmt &body,
                                    const Array<Stmt> &prologue_stmts,
                                    const Array<Var> &initial_defs) {
  if (prologue_stmts.empty()) {
    return body;
  }
  DependencyAwarePrologueInserter inserter(MakeSeqOrSingle(prologue_stmts),
                                           initial_defs);
  return inserter.Insert(body);
}

} // namespace

class LowerHopperIntrin : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f, bool disable_shuffle_elect) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    LowerHopperIntrin substituter(disable_shuffle_elect);
    fptr->body = substituter.VisitStmt(f->body);
    Map<Var, Array<PrimExpr>> init_desc_arg_map;
    // Collect prologue/epilogue statements for host-side setup/teardown
    Array<Stmt> prologue_stmts;
    Array<Stmt> epilogue_stmts;
    for (const auto &[call, var] : substituter.desc_map_) {
      // Should allocate 128 bytes for TensorMap on stack
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
      // add to function attribute
      Call init_desc =
          Call(DataType::Handle(), builtin::tvm_call_packed(), init_desc_args);
      // Accumulate TMA descriptor init into prologue
      prologue_stmts.push_back(LetStmt(var, alloc_desc, Evaluate(init_desc)));
      init_desc_arg_map.Set(var, init_desc_args);
    }
    f = WithAttr(std::move(f), "tma_descriptor_args", init_desc_arg_map);

    // Additionally, if L2 persistent cache annotations were lowered earlier,
    // materialize TVM FFI calls to set the stream access policy window.
    if (f->attrs.defined() && f->attrs->dict.count("l2_persistent_map")) {
      auto l2_map =
          f->GetAttr<Map<String, Array<PrimExpr>>>("l2_persistent_map");
      if (l2_map.defined()) {
        // Build a lookup from buffer name to Buffer object
        std::unordered_map<std::string, Buffer> name2buf;
        for (const auto &kv : f->buffer_map) {
          name2buf.emplace(kv.second->name, kv.second);
        }
        for (const auto &kv : l2_map.value()) {
          const std::string buf_name = kv.first;
          const Array<PrimExpr> &args = kv.second;
          if (name2buf.count(buf_name) == 0) {
            continue;
          }
          const Buffer &buf = name2buf.at(buf_name);
          // Build base pointer expression.
          //
          // We only need the base address for CUDA stream access policy window
          // configuration. Using `Buffer::access_ptr` would materialize a
          // typed pointer cast based on `buf->dtype` (e.g. float16 -> `half*`)
          // in the generated C host stubs, which then requires a `half`
          // definition during host compilation. Since the runtime API treats
          // the pointer as opaque, keep it as `void*`/handle and adjust by
          // `elem_offset` in bytes when needed.
          PrimExpr base_ptr = buf->data;
          if (buf->elem_offset.defined() && !is_zero(buf->elem_offset)) {
            PrimExpr byte_offset =
                buf->elem_offset *
                IntImm(buf->elem_offset.dtype(), buf->dtype.bytes());
            base_ptr =
                Call(DataType::Handle(), builtin::handle_add_byte_offset(),
                     {base_ptr, byte_offset});
          }
          // Args packed: func_name, base_ptr, num_bytes, hit_ratio
          Array<PrimExpr> packed_args;
          packed_args.push_back(
              StringImm(tvm_cuda_stream_set_access_policy_window));
          packed_args.push_back(base_ptr);
          // size_in_bytes (args[1]) then hit_ratio (args[0])
          ICHECK_GE(args.size(), 2);
          packed_args.push_back(args[1]);
          packed_args.push_back(args[0]);
          prologue_stmts.push_back(Evaluate(Call(
              DataType::Int(32), builtin::tvm_call_packed(), packed_args)));
        }
        // Add a single epilogue call to reset the access policy window and
        // restore L2 limit
        Array<PrimExpr> reset_args;
        reset_args.push_back(
            StringImm(tvm_cuda_stream_reset_access_policy_window));
        epilogue_stmts.push_back(Evaluate(
            Call(DataType::Int(32), builtin::tvm_call_packed(), reset_args)));
      }
    }

    // Stitch prologue statements before the original body
    if (!prologue_stmts.empty()) {
      fptr->body = InsertAfterRequiredDefinitions(fptr->body, prologue_stmts,
                                                  GetPrologueInitialDefs(f));
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
    } else {
      return StmtExprMutator::VisitExpr_(call);
    }
  }

private:
  Array<Stmt> prefetch_calls_;
  std::unordered_map<Call, Var, StructuralHash, ExprDeepEqual> desc_map_;
  LowerHopperIntrin(bool disable_shuffle_elect)
      : disable_shuffle_elect_(disable_shuffle_elect) {}
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
