#include "preprocess_ir.h"
#include "compat/metadata.h"
#include "compat/region.h"

#include "compat/visit.h"
#include "cuda/transform/ws_analysis.h"
#include "op/builtin.h"
#include "op/utils.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>

#include <algorithm>
#include <map>
#include <set>

namespace tvm::tl::cuda {
namespace {
using namespace ffi;
using namespace tirx;

bool CheckCalls(const Stmt &body) {
  bool supported = true;
  PostOrderVisit(body, [&](const ObjectRef &object) {
    if (object.as<BreakNode>() || object.as<ContinueNode>() ||
        object.as<ReturnNode>()) {
      LOG(WARNING) << "AutoWarpSpecialization skipped: control-flow exits "
                      "require an explicit schedule";
      supported = false;
    }
    if (const auto *call = object.as<CallNode>()) {
      auto operation = call->op.as<Op>();
      if (IsBarrierOrTmaControlCall(call) ||
          (operation &&
           (operation.value()->name.find("atomic") != std::string::npos ||
            operation.value()->name.starts_with("tirx.ptx.cp_async_") ||
            operation.value()->name == "tl.loop_break"))) {
        LOG(WARNING) << "AutoWarpSpecialization skipped: hand-managed "
                        "synchronization or atomic operation "
                     << call->op;
        supported = false;
      }
    }
  });
  return supported;
}

bool CheckOpaque(const Stmt &statement, const Target &target) {
  bool supported = true;
  PostOrderVisit(statement, [&](const ObjectRef &object) {
    const auto *call = object.as<CallNode>();
    if (!call)
      return;
    auto operation = ParseOperator(GetRef<Call>(call));
    if (!operation.defined())
      return;
    if (const auto *copy = operation.as<CopyNode>())
      supported &= ClassifyCopy(copy, target) != TileStmtKind::kTmaProducer;
    if (auto gemm = GetGemmInfo(operation))
      supported &= !IsTmemBuffer(gemm->accumulator) && gemm->wg_wait == 0;
  });
  if (!supported)
    LOG(WARNING) << "AutoWarpSpecialization skipped: asynchronous operation "
                    "nested inside an indivisible statement";
  return supported;
}

class OpIdNormalizer : public tvm::tl::StmtExprMutator {
public:
  explicit OpIdNormalizer(Target target) : target_(std::move(target)) {}
  bool supported = true;

  void CollectIds(const Stmt &statement) {
    PostOrderVisit(statement, [&](const ObjectRef &object) {
      if (auto child = object.as<Stmt>())
        if (auto identifier = GetWSOpId(*child))
          used_ids_.insert(*identifier);
    });
  }

private:
  String FreshId(std::string kind) {
    std::replace(kind.begin(), kind.end(), '.', '_');
    while (true) {
      String identifier(kind + "_" + std::to_string(next_id_[kind]++));
      if (used_ids_.insert(identifier).second)
        return identifier;
    }
  }

  Stmt Wrap(const Stmt &statement, const std::string &kind) {
    return RegionStmt(::tvm::tl::scope_op(), {}, {},
                      DictAttrs({{kWSOpIdKey, FreshId(kind)}}),
                      SeqStmt(statement), {}, statement->loc);
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode) final {
    if (node->var->ty.as<TensorTypeNode>() || GetWSOpId(GetRef<Stmt>(node)))
      return Unchanged();
    DictAttrs attributes({{kWSOpIdKey, FreshId(std::string(node->var->name))}});
    Expr value = Call(node->value->ty, Op::Get("tl.ws_value"), {node->value},
                      attributes, {}, node->value->loc);
    return Bind(node->var, value, node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode *node, InplaceMode) final {
    const auto *call = node->value.as<CallNode>();
    if (!call)
      return Wrap(GetRef<Stmt>(node), "op");
    std::string name = "op";
    if (auto operation = call->op.as<Op>())
      name = operation.value()->name;
    auto identifier = GetWSOpId(GetRef<Stmt>(node));
    if (identifier && name != "tl.ptx_cp_async")
      return Unchanged();
    Map<String, Any> attributes;
    if (call->attrs) {
      auto dictionary = call->attrs.value().as<DictAttrs>();
      if (!dictionary)
        return Wrap(GetRef<Stmt>(node), name);
      attributes = dictionary.value()->dict;
    }
    if (!identifier)
      attributes.Set(kWSOpIdKey, FreshId(name));
    if (name == "tl.ptx_cp_async")
      attributes.Set("tl.ws_managed_cp_async", true);
    return Evaluate(Call(call->ty, call->op, call->args, DictAttrs(attributes),
                         call->ty_args, call->loc),
                    node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node, InplaceMode) final {
    std::string name = "store";
    if (auto variable = node->dest.as<Var>())
      name = variable.value()->name;
    return Wrap(GetRef<Stmt>(node), name);
  }

  UnchangedOr<Stmt> Mutate_(const IfNode *node, InplaceMode mode) final {
    Stmt statement = GetRef<Stmt>(node);
    bool contains_pipeline = false;
    PostOrderVisit(statement, [&](const ObjectRef &object) {
      if (const auto *loop = object.as<ForNode>())
        contains_pipeline |= loop->annotations.count("num_stages") != 0;
      if (auto nested = object.as<Evaluate>()) {
        auto kind = ClassifyStmt(*nested, target_);
        contains_pipeline |= kind == TileStmtKind::kTmaProducer ||
                             kind == TileStmtKind::kCpAsyncProducer ||
                             kind == TileStmtKind::kCpAsyncRaw ||
                             kind == TileStmtKind::kTcgen05Mma;
      }
    });
    if (contains_pipeline)
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    supported &= CheckOpaque(statement, target_);
    return Wrap(statement, "if");
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    Stmt statement = GetRef<Stmt>(node);
    if (GetWSOpId(statement)) {
      if (node->body->seq.size() == 1 && node->body->seq[0].as<WhileNode>()) {
        While loop = node->body->seq[0].as_or_throw<While>();
        SeqStmt body = Mutate(loop->body, mode).ValueOrUnchanged(loop->body);
        loop.CopyOnWrite()->body = body;
        RegionStmt region = GetRef<RegionStmt>(node);
        region.CopyOnWrite()->body = SeqStmt(loop, node->body->loc);
        return region;
      }
      supported &= CheckOpaque(statement, target_);
      return Unchanged();
    }
    supported &= CheckOpaque(statement, target_);
    RegionStmt region = GetRef<RegionStmt>(node);
    auto attributes = node->attrs->dict;
    attributes.Set(kWSOpIdKey, FreshId("scope"));
    region.CopyOnWrite()->attrs = DictAttrs(attributes);
    return region;
  }

  UnchangedOr<Stmt> Mutate_(const WhileNode *node, InplaceMode mode) final {
    While loop = GetRef<While>(node);
    SeqStmt body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
    loop.CopyOnWrite()->body = body;
    return Wrap(loop, "while");
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *node, InplaceMode mode) final {
    For loop = GetRef<For>(node);
    bool sequential =
        node->kind == ForKind::kDefault || node->kind == ForKind::kUnrolled;
    if (sequential) {
      SeqStmt body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
      loop.CopyOnWrite()->body = body;
    } else {
      supported &= CheckOpaque(loop, target_);
    }
    if (!node->annotations.count(kWSOpIdKey))
      loop.CopyOnWrite()->annotations.Set(
          kWSOpIdKey, FreshId(node->kind == ForKind::kParallel     ? "parallel"
                              : node->kind == ForKind::kVectorized ? "vectorize"
                              : node->kind == ForKind::kUnrolled   ? "unroll"
                                                                   : "loop"));
    return loop;
  }

  Target target_;
  std::set<String> used_ids_;
  std::map<std::string, int> next_id_;
};

Type WSValueType(const CallNode *call) {
  TVM_FFI_CHECK(call->args.size() == 1, ValueError)
      << "A scheduled scalar binding needs exactly one value";
  return call->args[0]->ty;
}
} // namespace

ffi::Optional<Stmt> PreprocessIR(Stmt body, const Target &target) {
  if (!CheckCalls(body))
    return std::nullopt;
  auto normalizer = ffi::make_object<OpIdNormalizer>(target);
  normalizer->CollectIds(body);
  Stmt result = normalizer->Mutate(body).ValueOrUnchanged(body);
  return normalizer->supported ? ffi::Optional<Stmt>(result) : std::nullopt;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterBuiltin<1>("tl.ws_value", CallEffectKind::kPure);
  OpDef("tl.ws_value")
      .set_native_fn_attr<&WSValueType>(tvm::op_attr::kInferType);
  ffi::reflection::GlobalDef()
      .def("tl.cuda.analysis.PreprocessIR", PreprocessIR)
      .def("tl.cuda.analysis.GetWSOpId", GetWSOpId);
}
} // namespace tvm::tl::cuda
