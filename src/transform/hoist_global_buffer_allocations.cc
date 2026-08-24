/*!
 * \brief Hoist global buffer allocations to the top of the block (host side).
 * \file hoist_global_buffer_allocations.cc
 */

#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/var.h>

#include "../op/utils.h"
#include "common/attr.h"
#include "tir/transforms/ir_utils.h"
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

class GlobalBufferAllocationsHoister : public StmtMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    auto node = Downcast<SBlock>(StmtMutator::VisitStmt_(op));

    if (IsHostMainBlock(op)) {
      for (const auto &buf : global_buffers_) {
        node.CopyOnWrite()->alloc_buffers.push_back(buf);
      }
    } else {
      ffi::Array<Buffer> new_alloc_buffers;
      for (const auto &buf : op->alloc_buffers) {
        if (IsGlobalBuffer(buf)) {
          global_buffers_.push_back(buf);
        } else {
          new_alloc_buffers.push_back(buf);
        }
      }
      node.CopyOnWrite()->alloc_buffers = std::move(new_alloc_buffers);
    }

    return node;
  }

  ffi::Array<Buffer> global_buffers_;
};

PrimFunc HoistGlobalBufferAllocations(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  GlobalBufferAllocationsHoister hoister;
  fptr->body = hoister(fptr->body);
  return func;
}

namespace transform {

Pass HoistGlobalBufferAllocations() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return ::tvm::tl::HoistGlobalBufferAllocations(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.HoistGlobalBufferAllocations",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.HoistGlobalBufferAllocations",
                        HoistGlobalBufferAllocations);
}

} // namespace transform

} // namespace tl
} // namespace tvm
