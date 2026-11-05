/*!
 * \file verify_buffer_init.cc
 * \brief Warn about reads of native tensors that nothing initializes.
 *
 * This is a permissive diagnostic, not a definite-assignment proof. Any
 * earlier write establishes per-thread storage. Shared storage accepts another
 * writer anywhere in the function, because a warp-specialized producer can
 * follow its consumer in source order. A read-modify-write operation must
 * never vouch for its own initial value.
 */

#include "ir/annotations.h"
#include "op/builtin.h"
#include "op/gemm.h"
#include "op/gemm_sp.h"
#include "op/operator.h"
#include "span_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <exception>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using tirx::StmtExprVisitor;
using tirx::TensorVar;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

bool IsExemptScope(const String &scope) {
  return scope == "global" || scope == "shared.barrier" ||
         scope == "shared.cluster_barrier";
}

bool IsCrossThreadScope(const String &scope) {
  return scope.find("shared") == 0;
}

String OpName(const CallObj *call) {
  auto op = call->op.as<Op>();
  return op.has_value() ? op.value()->name : String();
}

TileOperator TryParseOperator(const Call &call) {
  try {
    return ParseOperator(call);
  } catch (const std::exception &) {
    // Incomplete tile ops degrade to opaque here. Their later validator or
    // lowering, not this warning, diagnoses malformed IR.
    return TileOperator();
  }
}

/*! \brief Map tensor views and pointer aliases to backing storage identity.
 * A view does not initialize memory. Like the old data-Var model, a write to
 * any part of an allocation satisfies this coarse diagnostic for its views.
 * This does not assert full-region or path-sensitive coverage.
 */
class StorageAliases : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;

  Var Root(const Var &var) const {
    auto it = roots_.find(var);
    return it == roots_.end() ? var : it->second;
  }

  Optional<Var> Resolve(const Expr &expr) const {
    if (auto var = expr.as<Var>()) {
      if (var.value()->ty.as<tirx::TensorTypeObj>() ||
          var.value()->ty.as<PtrTypeObj>())
        return Root(var.value());
      return std::nullopt;
    }
    if (const auto *load = expr.as<TensorLoadObj>())
      return Resolve(load->source);
    if (const auto *region = expr.as<TensorRegionObj>())
      return Resolve(region->source);
    if (const auto *call = expr.as<CallObj>()) {
      const String name = OpName(call);
      if (!call->args.empty() &&
          (name == "tirx.tensor_data_ptr" || name == "tirx.decl_tensor" ||
           name == "tirx.reinterpret" || name == "tirx.ptr_byte_offset" ||
           name == "tirx.handle_add_byte_offset" || name == "tirx.address_of" ||
           name == "tl.access_ptr" || name == "tl.ptr_access"))
        return Resolve(call->args[0]);
    }
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const BindObj *op) final {
    if (op->var->ty.as<tirx::TensorTypeObj>() || op->var->ty.as<PtrTypeObj>()) {
      if (auto root = Resolve(op->value))
        roots_.insert_or_assign(op->var, root.value());
    }
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (OpName(op) == "tl.annotation")
      return std::nullopt;
    return StmtExprVisitor::Visit_(op);
  }

private:
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> roots_;
};

bool EscapesThroughArgs(const CallObj *call) {
  const String name = OpName(call);
  if (name == "tl.annotation" || name == "tirx.alloc_tensor" ||
      name == "tirx.decl_tensor")
    return false;
  if (name == "tl.access_ptr" || name == "tl.ptr_access" ||
      name == "tirx.address_of" || name.find("tl.tileop.") == 0)
    return true;
  auto op = call->op.as<Op>();
  if (!op.has_value())
    return true;
  const auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
  return !effects.count(op.value()) ||
         effects[op.value()] >= CallEffectKind::kUpdateState;
}

template <typename F>
void ForEachOpaqueWrite(const CallObj *call, const StorageAliases &aliases,
                        const F &record) {
  const String name = OpName(call);
  // Explicit read-only pointer intervals are not writes, even inside extern.
  if (name == "tl.access_ptr" || name == "tl.ptr_access") {
    const size_t mask_index = name == "tl.access_ptr" ? 2 : 3;
    if (call->args.size() > mask_index &&
        (GetConservativeAccessMask(
             call->args[mask_index].as_or_throw<PrimExpr>()) &
         kAccessWrite)) {
      if (auto root = aliases.Resolve(call->args[0]))
        record(root.value());
    }
    return;
  }
  for (const Expr &arg : call->args) {
    if (const auto *nested = arg.as<CallObj>()) {
      ForEachOpaqueWrite(nested, aliases, record);
    } else if (const auto *tuple = arg.as<TupleObj>()) {
      for (const Expr &item : tuple->fields)
        if (auto root = aliases.Resolve(item))
          record(root.value());
    } else if (auto root = aliases.Resolve(arg)) {
      record(root.value());
    }
  }
}

class PotentialWriteCollector : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  explicit PotentialWriteCollector(const StorageAliases &aliases)
      : aliases_(aliases) {}

  std::unordered_map<const VarObj *, std::vector<const Object *>> writers;

  void Record(const Expr &expr, const Object *writer) {
    if (auto root = aliases_.Resolve(expr))
      writers[root.value().get()].push_back(writer);
  }

  Optional<VisitInterrupt> Visit_(const TensorStoreObj *op) final {
    Record(op->dest, op);
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (OpName(op) == "tl.annotation" || OpName(op) == "tirx.decl_tensor" ||
        OpName(op) == "tirx.alloc_tensor")
      return std::nullopt;
    if (auto tile_op = TryParseOperator(GetRef<Call>(op)); tile_op.defined()) {
      for (const TensorRegion &region : tile_op->GetAccessRegions().writes)
        Record(region->source, op);
      return std::nullopt;
    }
    if (EscapesThroughArgs(op)) {
      ForEachOpaqueWrite(op, aliases_,
                         [&](const Var &var) { Record(var, op); });
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }

private:
  const StorageAliases &aliases_;
};

class BufferInitVerifier : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit;
  using StmtExprVisitor::Visit_;
  enum class Kind { kGeneric, kGemmAccum };
  struct Report {
    TensorVar tensor;
    Location loc;
    Kind kind;
  };

  explicit BufferInitVerifier(const tirx::Function &function)
      : aliases_(make_object<StorageAliases>()) {
    aliases_->Visit(function->body);
    for (const Var &param : function->params) {
      if (auto root = aliases_->Resolve(param)) {
        written_.insert(root.value());
        params_.insert(root.value());
      }
    }
    auto collector = make_object<PotentialWriteCollector>(*aliases_);
    collector->Visit(function->body);
    potential_writers_ = std::move(collector->writers);
  }

  Optional<VisitInterrupt> Visit(AnyView value) final {
    Location saved = current_loc_;
    if (const auto *stmt = value.as<StmtObj>()) {
      if (!stmt->loc.as<UnknownLocObj>())
        current_loc_ = stmt->loc;
    } else if (const auto *expr = value.as<ExprObj>()) {
      if (!expr->loc.as<UnknownLocObj>())
        current_loc_ = expr->loc;
    }
    auto result = StmtExprVisitor::Visit(value);
    current_loc_ = saved;
    return result;
  }

  bool WrittenByAnotherNode(const Var &var, const Object *reader) const {
    auto it = potential_writers_.find(var.get());
    if (it == potential_writers_.end())
      return false;
    for (const Object *writer : it->second)
      if (writer != reader)
        return true;
    return false;
  }

  void CheckRead(const Expr &source, Kind kind = Kind::kGeneric,
                 const Object *reader = nullptr) {
    auto tensor = source.as<TensorVar>();
    if (!tensor.has_value() || IsExemptScope(tensor.value().scope().name()))
      return;
    const Var root = aliases_->Root(tensor.value());
    if (reported_.count(root))
      return;
    const bool satisfied = params_.count(root) ||
                           (IsCrossThreadScope(tensor.value().scope().name())
                                ? WrittenByAnotherNode(root, reader)
                                : written_.count(root));
    if (!satisfied) {
      reported_.insert(root);
      reports_.push_back({tensor.value(), current_loc_, kind});
    }
  }

  Optional<VisitInterrupt> Visit_(const TensorStoreObj *op) final {
    const Object *saved = active_store_;
    active_store_ = op;
    auto result = StmtExprVisitor::Visit_(op);
    active_store_ = saved;
    if (auto root = aliases_->Resolve(op->dest))
      written_.insert(root.value());
    return result;
  }

  Optional<VisitInterrupt> Visit_(const TensorLoadObj *op) final {
    CheckRead(op->source, Kind::kGeneric, active_store_);
    return StmtExprVisitor::Visit_(op);
  }

  Optional<VisitInterrupt> Visit_(const CallObj *op) final {
    if (OpName(op) == "tl.annotation" || OpName(op) == "tirx.decl_tensor" ||
        OpName(op) == "tirx.alloc_tensor")
      return std::nullopt;
    if (auto tile_op = TryParseOperator(GetRef<Call>(op)); tile_op.defined()) {
      Optional<Expr> accum;
      if (const auto *gemm = tile_op.as<GemmNode>())
        accum = gemm->cRegion_->source;
      else if (const auto *gemm = tile_op.as<GemmSPNode>())
        accum = gemm->cRegion_->source;
      for (const TensorRegion &region : tile_op->GetReadBeforeWriteRegions()) {
        Kind kind = accum.has_value() && region->source.same_as(accum.value())
                        ? Kind::kGemmAccum
                        : Kind::kGeneric;
        CheckRead(region->source, kind, op);
      }
      for (const TensorRegion &region : tile_op->GetAccessRegions().writes)
        if (auto root = aliases_->Resolve(region->source))
          written_.insert(root.value());
      return std::nullopt;
    }
    if (EscapesThroughArgs(op)) {
      ForEachOpaqueWrite(op, *aliases_,
                         [&](const Var &var) { written_.insert(var); });
      return std::nullopt;
    }
    return StmtExprVisitor::Visit_(op);
  }

  void EmitReport() const {
    if (reports_.empty())
      return;
    std::ostringstream os;
    os << "Buffer read before initialization: " << reports_.size()
       << " buffer(s) read before anything writes them\n";
    for (size_t k = 0; k < reports_.size(); ++k) {
      const Report &report = reports_[k];
      os << "  [" << k + 1 << "] `" << report.tensor->name
         << "` is read before anything writes it"
         << SpanHintSuffix({report.loc}) << "\n";
      if (report.kind == Kind::kGemmAccum)
        os << "      T.gemm accumulates into C, so an uninitialized "
              "accumulator adds into stale contents. Add `T.clear(C)` before "
              "the gemm, or pass `clear_accum=True`.\n";
    }
    os << "Initialize the buffer before its first read (for example with "
          "`T.clear`, `T.fill`, or a `T.copy` into it). If you believe this "
          "is a false positive, disable the check by setting "
          "`PassConfigKey.TL_DISABLE_BUFFER_INIT_CHECK` in the pass config.";
    LOG(WARNING) << os.str();
  }

private:
  ObjectPtr<StorageAliases> aliases_;
  VarSet written_, params_, reported_;
  std::unordered_map<const VarObj *, std::vector<const Object *>>
      potential_writers_;
  std::vector<Report> reports_;
  Location current_loc_{UnknownLoc()};
  const Object *active_store_ = nullptr;
};

tvm::transform::Pass VerifyBufferInit() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext ctx) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) ||
        ctx->GetConfig<bool>(kDisableBufferInitCheck).value_or(false))
      return function;
    auto verifier = make_object<BufferInitVerifier>(function);
    verifier->Visit(function->body);
    verifier->EmitReport();
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.VerifyBufferInit");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.VerifyBufferInit",
                                   VerifyBufferInit);
}
} // namespace
} // namespace tvm::tl
