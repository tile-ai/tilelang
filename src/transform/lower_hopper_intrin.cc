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
          // Build base pointer expression (read access)
          PrimExpr base_ptr = buf.access_ptr(1);
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
      // Chain the Let/Evaluate statements sequentially
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
    // Insert the prefetch TMA descriptor statement TO the beginning of the
    // kernel
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        auto body = StmtExprMutator::VisitStmt(op->body);
        if (prefetch_calls_.empty() && init_mbarrier_calls_.empty()) {
          return AttrStmt(op->node, op->attr_key, op->value, body);
        } else {
          Array<Stmt> stmt_seq;
          if (!init_mbarrier_calls_.empty()) {
            auto alloc_mbarrier =
                Evaluate(Call(DataType::Handle(), builtin::create_barriers(),
                              {static_cast<int>(init_mbarrier_calls_.size())}));
            stmt_seq.push_back(alloc_mbarrier);
          }

          auto stmts = prefetch_calls_;
          stmts.insert(stmts.end(), init_mbarrier_calls_.begin(),
                       init_mbarrier_calls_.end());
          PrimExpr condition;
          if (!disable_shuffle_elect_) {
            condition = Call(DataType::Bool(), tl_shuffle_elect(), {0});
          } else {
            condition = EQ(iv->var, 0);
          }
          auto stmt_ = IfThenElse(condition,
                                  stmts.size() > 1 ? SeqStmt(stmts) : stmts[0]);
          stmt_seq.push_back(stmt_);
          if (!init_mbarrier_calls_.empty()) {
            // Note from FlashAttention:
            // Helps with visibility of barrier init operations across warps /
            // cta / cluster Available as a separate function so as to batch
            // inits across barriers and fence once Note : It must be composed
            // with an appropriate sync instruction with the right scope to
            // ensure visibility eg. __syncthreads() or a cluster_arrive() +
            // cluster_wait()
            Stmt mem_fence = Evaluate(Call(
                DataType::Handle(), tvm::tl::ptx_fence_barrier_init(), {}));
            stmt_seq.push_back(mem_fence);
            Stmt mem_sync =
                Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                              {StringImm("shared")}));
            stmt_seq.push_back(mem_sync);
          }
          stmt_seq.push_back(body);

          prefetch_calls_.clear();
          init_mbarrier_calls_.clear();
          return AttrStmt(op->node, op->attr_key, op->value, SeqStmt(stmt_seq));
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
    } else if (call->op.same_as(create_list_of_mbarrier())) {
      ICHECK(init_mbarrier_calls_.empty());
      int num_barriers = static_cast<int>(call->args.size());
      for (int i = 0; i < num_barriers; i++) {
        PrimExpr mbarrier = Call(DataType::Handle(), get_mbarrier(), {i});
        init_mbarrier_calls_.push_back(Evaluate(
            Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
                 {mbarrier, call->args[i]})));
      }
      return 0;
    } else {
      return StmtExprMutator::VisitExpr_(call);
    }
  }

private:
  Array<Stmt> prefetch_calls_;
  Array<Stmt> init_mbarrier_calls_;
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
