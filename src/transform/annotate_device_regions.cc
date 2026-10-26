#include "compat/metadata.h"
/*!
 * \file annotate_device_regions.cc
 * \brief Split device function from host.
 */
#include "common/attr.h"

#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {

class DeviceRegionAnnotater : public tvm::tl::StmtExprMutator {
public:
  explicit DeviceRegionAnnotater(Target target) : target_(std::move(target)) {}

private:
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    if (node->op->name == "tirx.device_scope") {
      if (node->attrs->dict.count("target")) {
        return ffi::Unchanged();
      }
      auto attributes = node->attrs->dict;
      attributes.Set("target", target_);
      return RegionStmt(node->op, node->args, node->body_params,
                        DictAttrs(attributes), node->body, node->result_vars,
                        node->loc);
    }
    if (node->op.same_as(tirx::launch_thread_op()) ||
        node->attrs->dict.count(attr::pipeline_exec_scope)) {
      return RegionStmt(Op::Get("tirx.device_scope"), {}, {},
                        DictAttrs({{"target", target_}}),
                        SeqStmt(ffi::GetRef<Stmt>(node)), {}, node->loc);
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  Target target_;
};

} // namespace

tvm::transform::Pass AnnotateDeviceRegions() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    auto target = function->GetAttr<Target>("target");
    TVM_FFI_CHECK(target.has_value(), ValueError)
        << "AnnotateDeviceRegions requires a target";
    if (target.value()->GetHost() && function->body) {
      Target device_target = target.value().WithoutHost();
      SeqStmt body = function->body.value();
      if (device_target->kind->name == "c") {
        const auto *region =
            body->seq.size() == 1 ? body->seq[0].as<RegionStmtNode>() : nullptr;
        if (region && region->op->name == "tirx.device_scope") {
          auto annotater =
              ffi::make_object<DeviceRegionAnnotater>(device_target);
          body = annotater->Mutate(body).ValueOrUnchanged(body);
        } else {
          body =
              SeqStmt(RegionStmt(Op::Get("tirx.device_scope"), {}, {},
                                 DictAttrs({{"target", device_target}}), body));
        }
      } else {
        auto annotater = ffi::make_object<DeviceRegionAnnotater>(device_target);
        body = annotater->Mutate(body).ValueOrUnchanged(body);
      }
      function.CopyOnWrite()->body = body;
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.AnnotateDeviceRegions");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AnnotateDeviceRegions",
                                   AnnotateDeviceRegions);
}

} // namespace tvm::tl
