/*!
 *  \file lower_shared_barrier.cc
 *  \brief Convert shared.barrier buffers to plain shared + ptx init.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

class SharedBarrierRewriter : public StmtMutator {
public:

  static PrimFunc Rewrite(const PrimFunc &f) {
    SharedBarrierRewriter rewriter;
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:

  Stmt VisitStmt_(const BlockNode *op) final {
    LOG(INFO) << "BlockNode: " << op->seq.size();
    return StmtMutator::VisitStmt_(op);
  }

  Array<Stmt> MakeInitCalls(const Map<Buffer, Buffer> &old2new) {
    // 确定 tx 维：假设已知 var "threadIdx.x"
    Var tx("threadIdx.x");
    PrimExpr cond = (tx == 0);

    Map<Buffer, Buffer> rev = old2new;
    Array<Stmt> inits;
    for (const auto &kv : rev) {
      const Buffer &old_buf = kv.first;
      const Buffer &new_buf = kv.second;
      int64_t count = 1;
      for (PrimExpr d : old_buf->shape) {
        if (auto *int_imm = d.as<IntImmNode>()) {
          count *= int_imm->value;
        }
      }
      // 构造 T.call_extern("ptx.init_barrier_thread_count", ...)
    //   Stmt call = Evaluate(Call(DataType::Int(32), builtin::call_extern(),
    //                             {
    //                                 StringImm("ptx.init_barrier_thread_count"),
    //                                 BufferLoad(new_buf, {0}),
    //                                 PrimExpr(count),
    //                             }));
    //   inits.push_back(IfThenElse(cond, SeqStmt({call})));
    }
    return inits;
  }

  static void FindAndInsertAllocs(const Stmt &s,
                                  const Map<Buffer, Buffer> &old2new,
                                  Array<Stmt> *append_allocs) {
    // Array<Stmt> *flat = const_cast<Array<Stmt> *>(&s->body);
    // auto *seq = const_cast<SeqStmtNode *>(s.as<SeqStmtNode>());
    // ICHECK(seq) << "Must visit SeqStmt";

    //  for (const auto& kv : old2new) {
    //    append_allocs->push_back(
    //        AllocBuffer(kv.second, {}, {}, NullOpt, Span()));
    //  }
  }

  Stmt InsertInitBarrier(const Stmt &body, const Map<Buffer, Buffer> &old2new) {
    Array<Stmt> inits = MakeInitCalls(old2new);
    // 在 body 开头插入 if-then 列表
    Stmt new_body = inits.empty() ? body : SeqStmt::Flatten(inits, body);
    return new_body;
  }

  std::unordered_set<const BufferNode *> barriers_;
};

PrimFunc LowerSharedBarrier(const PrimFunc &f) {
  SharedBarrierRewriter rewriter(collector.targets);
  return rewriter.Rewrite(f);
}

namespace transform {
using namespace tir::transform;

tvm::transform::Pass LowerSharedBarrier() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return tl::LowerSharedBarrier(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerSharedBarrier", {});
}

TVM_REGISTER_GLOBAL("tl.transform.LowerSharedBarrier")
    .set_body_typed(LowerSharedBarrier);

} // namespace transform
} // namespace tl
} // namespace tvm
