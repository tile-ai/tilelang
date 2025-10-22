/*!
 * \file warp_specialized_pipeline.cc
 * \brief Warp specialized Pipeline for cuda GPU (sm90+)
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;

bool isGemm(const Stmt &stmt) {
  bool is_gemm = false;
  if (stmt.as<EvaluateNode>()) {
    auto call = Downcast<Evaluate>(stmt)->value.as<CallNode>();
    if (call && call->op.same_as(Op::Get("tir.call_extern"))) {
      if (call->args[0].as<StringImmNode>()) {
        std::string name = Downcast<StringImm>(call->args[0])->value;
        if (name.find("gemm") != std::string::npos) {
          is_gemm = true;
        }
      }
    }
  }
  return is_gemm;
}

bool isGemmSync(const Stmt &stmt) {
  bool is_gemm_sync = false;
  if (stmt.as<EvaluateNode>()) {
    auto call = Downcast<Evaluate>(stmt)->value.as<CallNode>();
    if (call && call->op.same_as(Op::Get("tir.call_extern"))) {
      if (call->args[0].as<StringImmNode>()) {
        std::string name = Downcast<StringImm>(call->args[0])->value;
        if (name.find("warpgroup_wait") != std::string::npos) {
          is_gemm_sync = true;
        }
      }
    }
  }
  return is_gemm_sync;
}

bool isArriveBarrier(const Stmt &stmt) {
  bool is_arrive_barrier = false;
  if (stmt.as<EvaluateNode>()) {
    auto call = Downcast<Evaluate>(stmt)->value.as<CallNode>();
    if (call && call->op.same_as(Op::Get("tir.ptx_arrive_barrier"))) {
      is_arrive_barrier = true;
    }
  }
  return is_arrive_barrier;
}

class WgmmaSyncRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    auto T = WgmmaSyncRewriter();
    T.buffer_lca_ = DetectBufferAccessLCA(f);
    for (auto [buffer, _] : T.buffer_lca_)
      T.buffer_data_to_buffer_.Set(buffer->data, buffer);
    f.CopyOnWrite()->body = T(f->body);
    return f;
  }

private:
  void CollectWgmmaInfo(const SeqStmtNode *op) {
    for (int i = 0; i < static_cast<int>(op->seq.size()); i++) {
      auto stmt = op->seq[i];
      if (isGemm(stmt)) {
        gemm_stmts_.push_back(stmt);
        gemm_stmt_ids_.push_back(i);
        bool found_release = false;
        for (int j = i + 1; j < static_cast<int>(op->seq.size()); j++) {
          auto release_stmt = op->seq[j];
          if (isArriveBarrier(release_stmt)) {
            found_release = true;
            gemm_release_stmts_.push_back(release_stmt);
            break;
          }
        }
        if (!found_release) {
          gemm_release_stmts_.push_back(Evaluate(0));
        }
        // ICHECK(op->seq.size() > i + 1);
        // auto release_stmt = op->seq[i + 1];
        // auto next_call =
        // Downcast<Evaluate>(release_stmt)->value.as<CallNode>();
        // ICHECK(next_call);
        // ICHECK(next_call->op.same_as(Op::Get("tir.ptx_arrive_barrier")));
        Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                    /*name_hint=*/"",
                    /*body*/ op->seq[i]);
        auto access = GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
        std::set<const BufferNode *> read_set, write_set;
        for (auto region : access[0])
          read_set.insert(region->buffer.get());
        for (auto region : access[1])
          write_set.insert(region->buffer.get());
        gemm_read_buffers_.push_back(read_set);
        gemm_write_buffers_.push_back(write_set);
      }
    }
  }

  Stmt VisitStmt_(const ForNode *op) final {
    auto order_anno = op->annotations.Get("tl_pipeline_order");
    if (!order_anno) {
      return StmtExprMutator::VisitStmt_(op);
    }

    CollectWgmmaInfo(op->body.as<SeqStmtNode>());
    auto stmt_node = (op->body).as<SeqStmtNode>();
    ICHECK(stmt_node);

    auto intersect_fn = [](const std::set<const BufferNode *> &lhs,
                           const std::set<const BufferNode *> &rhs) {
      for (auto ptr : lhs)
        if (rhs.count(ptr))
          return true;
      return false;
    };

    for (int r = 0; r < static_cast<int>(gemm_stmts_.size()); r++) {
      bool found = false;
      auto last_stmt = Stmt();
      for (int i = 0; i < static_cast<int>(stmt_node->seq.size()); i++) {
        if (stmt_node->seq[i].same_as(gemm_stmts_[r])) {
          found = true;
          last_stmt = stmt_node->seq[i];
          continue;
        }
        if (!found)
          continue;
        Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                    /*name_hint=*/"",
                    /*body*/ stmt_node->seq[i]);
        auto access = GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
        std::set<const BufferNode *> read_set, write_set;
        for (auto region : access[0])
          read_set.insert(region->buffer.get());
        for (auto region : access[1])
          write_set.insert(region->buffer.get());
        if (intersect_fn(read_set, gemm_write_buffers_[r]) ||
            intersect_fn(write_set, gemm_read_buffers_[r]) ||
            intersect_fn(write_set, gemm_write_buffers_[r])) {
          break;
        }
        last_stmt = stmt_node->seq[i];
      }
      last_stmts_.push_back(last_stmt);
    }

    auto new_seq = Array<Stmt>();
    for (int i = 0; i < static_cast<int>(stmt_node->seq.size()); i++) {
      bool remove_ = false;
      for (int j = 0; j < static_cast<int>(gemm_stmts_.size()); j++) {
        if (stmt_node->seq[i].same_as(gemm_release_stmts_[j])) {
          remove_ = true;
          continue;
        }
      }
      if (remove_)
        continue;
      auto stmt = stmt_node->seq[i];
      for (int j = 0; j < static_cast<int>(gemm_stmts_.size()); j++) {
        if (stmt_node->seq[i].same_as(gemm_stmts_[j])) {
          auto call = Downcast<Evaluate>(stmt)->value.as<CallNode>();
          ICHECK(call);
          ICHECK(call->op.same_as(Op::Get("tir.call_extern")));
          ICHECK(call->args[0].as<StringImmNode>());
          std::string name = Downcast<StringImm>(call->args[0])->value;
          std::string new_name = name.substr(0, name.size() - 1) + ", -1>";
          auto new_args = Array<PrimExpr>();
          new_args.push_back(StringImm(new_name));
          for (int k = 1; k < static_cast<int>(call->args.size()); k++) {
            new_args.push_back(call->args[k]);
          }
          stmt = Evaluate(
              Call(DataType::Handle(), builtin::call_extern(), new_args));
          break;
        }
      }

      new_seq.push_back(stmt);
      for (int j = 0; j < static_cast<int>(gemm_stmts_.size()); j++) {
        if (stmt_node->seq[i].same_as(last_stmts_[j])) {
          Array<PrimExpr> new_args;
          new_args.push_back(StringImm("cute::warpgroup_wait<0>"));
          new_args.push_back(Integer(j));
          auto new_call =
              Call(DataType::Handle(), builtin::call_extern(), new_args);
          new_seq.push_back(Evaluate(new_call));
          if (std::count(gemm_release_stmts_.begin(), gemm_release_stmts_.end(),
                         gemm_release_stmts_[j]) == 1) {
            new_seq.push_back(gemm_release_stmts_[j]);
          } else {
            gemm_release_stmts_[j] = Evaluate(0);
          }
        }
      }
    }

    int gemm_count = 0;
    int max_sync_index = 0;
    for (int i = 0; i < static_cast<int>(new_seq.size()); i++) {
      if (isGemm(new_seq[i])) {
        gemm_count++;
      } else if (isGemmSync(new_seq[i])) {
        auto call = Downcast<Evaluate>(new_seq[i])->value.as<CallNode>();
        auto sync_index =
            static_cast<int>(Downcast<IntImm>(call->args[1])->value);
        auto wait_count = gemm_count - sync_index - 1;
        if (sync_index > max_sync_index)
          max_sync_index = sync_index;
        if (sync_index < max_sync_index) {
          // new_seq.erase(new_seq.begin() + i);
          new_seq.Set(i, Evaluate(0));
        } else {
          Array<PrimExpr> new_args;
          std::string call_str =
              "cute::warpgroup_wait<" + std::to_string(wait_count) + ">";
          new_args.push_back(StringImm(call_str));
          new_seq.Set(i, Evaluate(Call(DataType::Handle(),
                                       builtin::call_extern(), new_args)));
        }
      }
    }
    auto new_for =
        For(op->loop_var, op->min, op->extent, op->kind,
            new_seq.size() == 1 ? new_seq[0] : SeqStmt(std::move(new_seq)),
            op->thread_binding, op->annotations);
    return new_for;
  }

  WgmmaSyncRewriter() = default;

  Map<Buffer, Optional<Stmt>> buffer_lca_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  std::vector<std::set<const BufferNode *>> gemm_read_buffers_;
  std::vector<std::set<const BufferNode *>> gemm_write_buffers_;
  std::vector<Stmt> gemm_stmts_;
  std::vector<Stmt> gemm_release_stmts_;
  std::vector<Stmt> last_stmts_;

  std::vector<int32_t> gemm_stmt_ids_;
  friend class WgmmaReleaseCollector;
};

using namespace tir::transform;

tvm::transform::Pass RewriteWgmmaSync() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return WgmmaSyncRewriter::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RewriteWgmmaSync", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RewriteWgmmaSync", RewriteWgmmaSync);
}

} // namespace tl
} // namespace tvm
