/*!
 * \file add_bufstore_wrapper.cc
 * \brief Wrap single buffer stores with parallel loops.
 */

#include "tvm/runtime/logging.h"
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

using VarSet = std::unordered_set<Var>;
using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief Helper function to calculate the intersection of two sets.
 * \tparam T The type of the elements.
 * \tparam Hash The hash function for the elements.
 * \tparam Equal The equality function for the elements.
 * \param left The first set.
 * \param right The second set.
 * \return The intersection of the two sets.
 */
template <typename T, typename Hash = std::hash<T>,
          typename Equal = std::equal_to<T>>
auto SetIntersection(const std::unordered_set<T, Hash, Equal> &left,
                     const std::unordered_set<T, Hash, Equal> &right) {
  std::unordered_set<T, Hash, Equal> result;
  // Iter the smaller set for better performance.
  const auto &smaller = left.size() <= right.size() ? left : right;
  const auto &larger = left.size() <= right.size() ? right : left;
  for (const auto &item : smaller)
    if (larger.find(item) != larger.end())
      result.insert(item);
  return result;
}

/*!
 * \brief Collect used variables in a statement.
 */
class UsedVarsCollector : StmtExprVisitor {
public:
  UsedVarsCollector() : used_variables({}) {}

  /*!
   * \brief Entry point to collect used variables.
   * \param stmt The statement to collect used variables.
   * \return The used variables.
   */
  static VarSet Collect(const Stmt &stmt) {
    UsedVarsCollector collector;
    collector.VisitStmt(stmt);
    return collector.used_variables;
  }

private:
  void VisitExpr_(const VarNode *op) final {
    used_variables.insert(ffi::GetRef<Var>(op));
  }

private:
  VarSet used_variables;
};

/*!
 * \brief Collect accessed buffers in a statement and categorizes them by their
 * scope.
 */
class AccessedBuffersCollector : public StmtExprVisitor {
public:
  AccessedBuffersCollector() = default;

  /*!
   * \brief Entry point to collect accessed buffers.
   * \param stmt The statement to collect accessed buffers.
   * \return The accessed buffers.
   */
  static std::pair<BufferSet, BufferSet> Collect(const Stmt &stmt) {
    AccessedBuffersCollector collector;
    collector.VisitStmt(stmt);
    return {collector.local_buffers, collector.fragment_buffers};
  }

private:
  /*!
   * \brief Categorize buffers by their scope.
   * \param buf The buffer to categorize.
   */
  inline void AddBuffer(const Buffer &buf) {
    if (buf.scope() == "local") {
      local_buffers.insert(buf);
    } else if (buf.scope() == "local.fragment") {
      fragment_buffers.insert(buf);
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    this->AddBuffer(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    this->AddBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  BufferSet local_buffers;
  BufferSet fragment_buffers;
};

using BufferIndicesMap = std::unordered_map<Buffer, std::vector<PrimExpr>,
                                            ObjectPtrHash, ObjectPtrEqual>;

/*!
 * \brief Collect buffer indices in a statement and return the map from buffer
 * to indices.
 */
class BufferIndicesCollector : public StmtExprVisitor {
public:
  BufferIndicesCollector() = default;

  /*!
   * \brief Entry point to collect buffer indices.
   * \param stmt The statement to collect buffer indices.
   * \return The map from buffer to indices.
   */
  static BufferIndicesMap Collect(const Stmt &stmt) {
    BufferIndicesCollector collector;
    collector.VisitStmt(stmt);
    return collector.buffer_indices;
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (buffer_indices.find(op->buffer) == buffer_indices.end())
      buffer_indices[op->buffer] = {};

    auto &vec = buffer_indices[op->buffer];
    for (const PrimExpr &idx : op->indices)
      vec.push_back(idx);

    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (buffer_indices.find(op->buffer) == buffer_indices.end())
      buffer_indices[op->buffer] = {};

    auto &vec = buffer_indices[op->buffer];
    for (const PrimExpr &idx : op->indices)
      vec.push_back(idx);

    StmtExprVisitor::VisitStmt_(op);
  }

private:
  BufferIndicesMap buffer_indices;
};

/*!
 * \brief Wrap single buffer stores with parallel loops.
 * \note This transformation adds T.Parallel wrappers around buffer stores that:
 *       1. Access fragment buffers with index 0
 *       2. Are not inside existing tile operations or thread bindings
 *       3. Don't access fragment buffers with non-zero indices
 */
class AddWrapperForSingleBufStore : public StmtMutator {
public:
  AddWrapperForSingleBufStore() = default;

  /*!
   * \brief Entry point to wrap single buffer stores.
   * \param func The function to wrap single buffer stores.
   * \return The function with wrapped single buffer stores.
   */
  static PrimFunc Apply(PrimFunc func) {
    AddWrapperForSingleBufStore wrapper;
    PrimFuncNode *func_node = func.CopyOnWrite();
    func_node->body = wrapper.VisitStmt(func_node->body);
    return func;
  }

private:
  /*!
   * \brief Check if the given for loop is a tile operation loop (parallel or
   * has num_stages annotation).
   * \param for_node The for loop to check.
   * \return True if the given for loop is a tile operation loop.
   */
  inline bool IsTileOperationLoop(const ForNode *for_node) {
    return for_node->kind == ForKind::kParallel ||
           for_node->annotations.find("num_stages") !=
               for_node->annotations.end();
  }

  /*!
   * \brief Check if the given attribute statement is a thread extent attribute.
   * \param attr_node The attribute statement to check.
   * \return True if the given attribute statement is a thread extent attribute.
   */
  inline bool IsThreadExtentAttr(const AttrStmtNode *attr_node) {
    return attr_node->attr_key == tir::attr::thread_extent;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    bool is_tile_operation_loop = IsTileOperationLoop(op);
    if (is_tile_operation_loop)
      ++tile_operation_depth;
    Stmt new_stmt = StmtMutator::VisitStmt_(op);
    if (is_tile_operation_loop)
      --tile_operation_depth;
    return new_stmt;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (IsThreadExtentAttr(op))
      thread_binding_vars.insert(
          ffi::GetRef<IterVar>(op->node.as<IterVarNode>())->var);
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    VarSet used_variables =
        UsedVarsCollector::Collect(ffi::GetRef<BufferStore>(op));
    VarSet thread_bound_variables =
        SetIntersection(used_variables, thread_binding_vars);
    // Only transform if not inside tile operations and no thread bindings
    if (tile_operation_depth == 0 && thread_bound_variables.empty()) {
      std::pair<BufferSet, BufferSet> pair =
          AccessedBuffersCollector::Collect(ffi::GetRef<BufferStore>(op));
      // Skip if no fragment buffers are accessed
      if (pair.second.empty())
        return ffi::GetRef<BufferStore>(op);
      // Validate fragment buffer indices - only index 0 is supported
      BufferIndicesMap buffer_indices =
          BufferIndicesCollector::Collect(ffi::GetRef<BufferStore>(op));
      for (const auto &[buf, indices] : buffer_indices) {
        if (buf.scope() != "local.fragment")
          continue;
        for (const auto &idx : indices) {
          bool invalid = false;
          if (const auto *imm = idx.as<IntImmNode>()) {
            invalid = imm->value != 0;
          }
          LOG_IF(FATAL, invalid)
              << "Fragment buffer access with non-zero index " << idx
              << " is not supported. Only fragment[0] access is allowed.";
        }
      }
      // Wrap fragment[0] access with T.Parallel loop
      return For(Var{"_"}, 0, 1, ForKind::kParallel,
                 ffi::GetRef<BufferStore>(op));
    }
    return ffi::GetRef<BufferStore>(op);
  }

private:
  /*! \brief Counter for tracking nested tile operations. */
  size_t tile_operation_depth = 0;
  /*! \brief Set of thread binding variables. */
  VarSet thread_binding_vars;
};

tvm::transform::Pass AddWrapperForSingleBufStorePass() {
  using namespace tir::transform;
  auto pass_func = [](PrimFunc f, const IRModule &, PassContext) {
    return AddWrapperForSingleBufStore::Apply(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AddWrapperForSingleBufStore", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AddWrapperForSingleBufStore",
                        AddWrapperForSingleBufStorePass);
}

} // namespace tl
} // namespace tvm
