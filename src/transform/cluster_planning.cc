/*!
 * \file clasuter_planning.cc
 * \brief Plan the cluster for GPU(sm90+) blocks
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../support/ffi_aliases.h"

namespace tvm {
namespace tir {

class ClusterPlanner {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    // Step 1: Collect the read region of the function
    Map<Var, Buffer> buffer_data_to_buffer_;
    for (const auto &[_, buffer] : f->buffer_map) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
                /*body*/ f->body);
    Array<Array<BufferRegion>> access =
        GetBlockReadWriteRegion(block, buffer_data_to_buffer_);
    auto reads = access[0];

    BlockIdxVisitor blockIdx_visitor;
    blockIdx_visitor(f->body);
    auto dom_map = blockIdx_visitor.dom_map_;

    // Step 2: Collect mem reuse count for clustering on each dimension.
    std::unordered_map<const IterVarNode *, size_t> mem_reuse_count;
    for (auto iv : dom_map)
      mem_reuse_count[iv] = 0;

    for (const auto &buffer_region : reads) {
      PrimExpr size = buffer_region->buffer->dtype.bits();
      RegionVisitor visitor;
      for (const auto &range : buffer_region->region) {
        size = size * range->extent;
        visitor(range->min);
      }
      size = arith::Analyzer().Simplify(size);
      if (auto imm = size.as<IntImmNode>()) {
        for (auto iv : dom_map) {
          if (visitor.seen_.count(iv->var.get()) == 0)
            mem_reuse_count[iv] += imm->value;
        }
      }
    }

    // Step 3: Pick the cluster dimension with the largest mem_reuse.
    size_t mem_reuse_max = 0;
    String cluster_tag;
    for (auto iv : dom_map) {
      if (auto extent = iv->dom->extent.as<IntImmNode>()) {
        if (extent->value % cluster_size_ == 0 &&
            mem_reuse_count[iv] > mem_reuse_max) {
          cluster_tag = iv->thread_tag;
          mem_reuse_max = mem_reuse_count[iv];
        }
      }
    }

    if (mem_reuse_max > 0) {
      std::string tag_str =
          static_cast<std::string>(cluster_tag); // Convert to std::string
      if (tag_str.rfind("blockIdx", 0) == 0) {
        // starts with "blockIdx"
        tag_str = "clusterIdx" + tag_str.substr(strlen("blockIdx"));
      } else {
        // Unexpected format — maybe just prefix
        tag_str = "clusterIdx" + tag_str;
      }
      cluster_tag = String(tag_str); // Convert back
      return WithAttr(f, cluster_tag, Integer(cluster_size_));
    } else {
      return f;
    }
  }

private:
  ClusterPlanner() = default;

  class RegionVisitor : public ExprVisitor {
  public:
    RegionVisitor() {};
    void VisitExpr_(const VarNode *var) { seen_.insert(var); }
    std::unordered_set<const VarNode *> seen_;
  };

  class BlockIdxVisitor : public StmtVisitor {
  public:
    BlockIdxVisitor() {};
    void VisitStmt_(const AttrStmtNode *attr) final {
      if (attr->attr_key == attr::thread_extent) {
        IterVar iv = Downcast<IterVar>(attr->node);
        String tag = iv->thread_tag;
        if (tag == "blockIdx.x" || tag == "blockIdx.y" || tag == "blockIdx.z")
          dom_map_.insert(iv.get());
      }
      StmtVisitor::VisitStmt_(attr);
    }
    /*! \brief The map from vars to blockidx extents. */
    std::unordered_set<const IterVarNode *> dom_map_;
  };

  /*! \brief Currently set the plossible cluster size as 2 */
  const static int cluster_size_ = 2;
};

PrimFunc ClusterPlanning(PrimFunc f) { return ClusterPlanner::Substitute(f); }

namespace transform {

tvm::transform::Pass ClusterPlanning() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return ClusterPlanning(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ClusterPlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ClusterPlanning", ClusterPlanning);
}
} // namespace transform

} // namespace tir
} // namespace tvm
