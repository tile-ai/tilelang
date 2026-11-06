/*!
 * \file normalize_conflict_hints.cc
 * \brief Normalize scheduling assertions without escaping native definitions.
 *
 * A direct hint keeps the B20 four-field contract at its original statement
 * site. Ancestor/root projections and grouped half-declarations instead keep
 * their own operands in a scoped metadata record. The scheduling analysis
 * pairs those records with source contexts; no TensorRegion, loop variable,
 * view type, or mutable definition is hoisted into an enclosing IR header.
 */

#include "ascend/transform/source_scopes.h"
#include "layout/expr_utils.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/scheduled_tir.h"
#include "ir/annotations.h"
#include "ir/loop_math.h"
#include "op/utils.h"
#include <tvm/sym/analyzer.h>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
using namespace ascend;
using namespace prim;

namespace {
constexpr const char *kMarker = "tl.conflict_hint";
constexpr const char *kDirect = "conflict_hint";
constexpr const char *kScoped = "tl.scoped_conflict_hints";
constexpr const char *kScopeId = "tl.conflict_scope_id";
constexpr int kAllScopes = -2;
constexpr int kRoot = -1;
constexpr int kNoDirectScope = -2;
using Fields = Map<String, Any>;

bool IsCallNamed(const CallObj *call, const char *name) {
  const auto *op = call ? call->op.as<OpObj>() : nullptr;
  return op && op->name == name;
}

struct Marker {
  Array<Expr> operands;
  int level;
  int cross;
  bool polarity;
  String group;
  String hint_id;
  String role{"pair"};
  std::vector<const ForObj *> chain;
  int direct_scope{kNoDirectScope};
  std::vector<const ForObj *> targets; // nullptr denotes this kernel root.
};

class ConflictHintNormalizer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static RegionStmt Rewrite(RegionStmt root) {
    auto rewriter = make_object<ConflictHintNormalizer>();
    rewriter->ReserveExistingIds(root->body);
    rewriter->Scan(root->body, kRoot);
    rewriter->ResolveGroups();
    if (rewriter->markers_.empty())
      return root;
    root.CopyOnWrite()->body =
        rewriter->Mutate(root->body).ValueOrUnchanged(root->body);
    TVM_FFI_ICHECK_EQ(rewriter->cursor_, rewriter->markers_.size());
    return root;
  }

private:
  std::vector<Marker> markers_;
  std::vector<const ForObj *> chain_;
  std::unordered_map<const ForObj *, String> scope_ids_;
  std::unordered_set<std::string> used_scopes_, used_hints_;
  size_t next_scope_{0}, next_hint_{0}, cursor_{0};

  String NewId(const char *prefix, size_t *next,
               std::unordered_set<std::string> *used) {
    std::string value;
    do {
      value = prefix + std::to_string((*next)++);
    } while (used->count(value));
    used->insert(value);
    return value;
  }

  void ReserveExistingIds(const Stmt &body) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        body, [&](const ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *loop = object.as<ForObj>()) {
            if (auto id = loop->annotations.Get(kScopeId)) {
              String name = id->cast<String>();
              scope_ids_.emplace(loop, name);
              used_scopes_.insert(std::string(name));
            }
          }
          if (const auto *call = object.as<CallObj>(); IsAnnotationCall(call)) {
            if (auto value = ReadAnnotations(call).Get(kScoped))
              for (const Any &entry : value->cast<Array<Any>>()) {
                auto fields = entry.cast<Fields>();
                used_hints_.insert(
                    std::string(fields.at("hint_id").cast<String>()));
              }
          }
          return ffi::WalkResult::Advance();
        });
  }

  static int ConstantInteger(const Expr &expr, const char *field) {
    auto value = expr.as<PrimExpr>();
    auto integer = value ? TryConstInt(*value) : std::nullopt;
    TVM_FFI_CHECK(integer && *integer >= std::numeric_limits<int32_t>::min() &&
                      *integer <= std::numeric_limits<int32_t>::max(),
                  ValueError)
        << "conflict-hint marker requires a constant int32 " << field;
    return static_cast<int>(*integer);
  }

  static Expr Operand(const Expr &value) {
    if (value.as<TensorRegionObj>())
      return value;
    if (const auto *var = value.as<VarObj>()) {
      TVM_FFI_CHECK(var->ty.as<TensorTypeObj>() || var->ty.as<PtrTypeObj>(),
                    ValueError)
          << "conflict-hint bare operand must be a native tensor or pointer";
      return value;
    }
    if (const auto *call = value.as<CallObj>();
        IsCallNamed(call, "tirx.tensor_data_ptr"))
      return call->args[0].as_or_throw<TensorVar>();
    return NormalizeToTensorRegion(value);
  }

  enum ScanExit : unsigned {
    kFallsThrough = 0,
    kBreaks = 1,
    kContinues = 2,
    kReturns = 4,
    kMayNotFinish = 8,
  };

  static unsigned AfterLoop(const ForObj *loop, unsigned body_exit) {
    sym::Analyzer analyzer;
    bool pure_header = SideEffect(loop->begin) <= CallEffectKind::kPure &&
                       SideEffect(loop->end) <= CallEffectKind::kPure &&
                       SideEffect(loop->step) <= CallEffectKind::kPure;
    // Native For always tests var < end, including when step is negative.
    // An empty loop does not execute a return in its body. Otherwise only a
    // positive, non-wrapping update proves that execution reaches the suffix.
    if (pure_header && analyzer->CanProve(loop->begin >= loop->end))
      return kFallsThrough;
    unsigned escaping = body_exit & (kReturns | kMayNotFinish);
    if (!CanUseLoopBounds(loop->begin, loop->end, loop->step, analyzer.get()))
      escaping |= kMayNotFinish;
    // Break and Continue belong to this loop. They restrict declarations in
    // its body, but do not restrict the next statement of an enclosing loop.
    return escaping;
  }

  unsigned Scan(const Stmt &stmt, int direct_scope) {
    if (const auto *seq = stmt.as<SeqStmtObj>()) {
      unsigned escaping = kFallsThrough;
      for (const Stmt &child : seq->seq) {
        // A direct list is unconditional in its target sequence. A jump in
        // any preceding branch makes a suffix declaration conditional, even
        // though the declaration itself is not syntactically inside an If.
        escaping |= Scan(child, escaping ? kNoDirectScope : direct_scope);
      }
      return escaping;
    } else if (const auto *loop = stmt.as<ForObj>()) {
      chain_.push_back(loop);
      unsigned escaping = Scan(loop->body, static_cast<int>(chain_.size()) - 1);
      chain_.pop_back();
      return AfterLoop(loop, escaping);
    } else if (const auto *branch = stmt.as<IfObj>()) {
      unsigned then_exit = Scan(branch->then_case, kNoDirectScope);
      unsigned else_exit = branch->else_case
                               ? Scan(*branch->else_case, kNoDirectScope)
                               : kFallsThrough;
      if (auto constant = TryConstInt(branch->condition))
        return *constant ? then_exit : else_exit;
      return then_exit | else_exit;
    } else if (const auto *loop = stmt.as<WhileObj>()) {
      unsigned escaping = Scan(loop->body, kNoDirectScope);
      if (auto constant = TryConstInt(loop->condition); constant && !*constant)
        return kFallsThrough;
      return (escaping & kReturns) | kMayNotFinish;
    } else if (const auto *scope = stmt.as<RegionStmtObj>()) {
      // Thread launch regions do not enter the counted loop chain. A dynamic
      // or atomic region retains its source context for the analysis index.
      auto attribute = ReadSourceAttribute(scope);
      bool transparent = attribute && attribute->key == "tl.loop.annotations";
      return Scan(scope->body, transparent ? direct_scope : kNoDirectScope);
    } else if (stmt.as<BreakObj>()) {
      return kBreaks;
    } else if (stmt.as<ContinueObj>()) {
      return kContinues;
    } else if (stmt.as<ReturnObj>()) {
      return kReturns;
    } else if (const auto *evaluate = stmt.as<EvaluateObj>()) {
      const auto *call = evaluate->value.as<CallObj>();
      if (!IsCallNamed(call, kMarker))
        return kFallsThrough;
      auto args = DecodeCallOperands(call);
      TVM_FFI_CHECK_EQ(args.size(), 6U, ValueError)
          << "conflict-hint marker expects 6 arguments";
      const auto polarity_type = args[5]->ty.as<PrimType>();
      TVM_FFI_CHECK(polarity_type && polarity_type.value() == PrimType::Bool(),
                    ValueError)
          << "conflict-hint marker expects a constant boolean polarity";
      int polarity = ConstantInteger(args[5], "polarity");
      int level = ConstantInteger(args[2], "level");
      int cross = ConstantInteger(args[3], "cross");
      TVM_FFI_CHECK(level >= kAllScopes, ValueError)
          << "conflict-hint level must be >= -2";
      TVM_FFI_CHECK(cross >= -1 && cross <= 1, ValueError)
          << "conflict-hint cross mode must be -1, 0 or 1";
      const auto *group = args[4].as<StringImmObj>();
      TVM_FFI_CHECK(group, ValueError)
          << "conflict-hint group must be a constant string";
      Marker marker;
      marker.operands = {Operand(args[0]), Operand(args[1])};
      marker.level = level;
      marker.cross = cross;
      marker.polarity = polarity != 0;
      marker.group = group->value;
      marker.chain = chain_;
      marker.direct_scope = direct_scope;
      markers_.push_back(std::move(marker));
    }
    return kFallsThrough;
  }

  static size_t CommonPrefix(const std::vector<const ForObj *> &a,
                             const std::vector<const ForObj *> &b) {
    size_t n = std::min(a.size(), b.size()), i = 0;
    while (i < n && a[i] == b[i])
      ++i;
    return i;
  }

  static std::vector<const ForObj *> Targets(const Marker &marker,
                                             size_t bound) {
    std::vector<const ForObj *> targets;
    if (marker.level == kRoot || marker.level == kAllScopes)
      targets.push_back(nullptr);
    if (marker.level == kAllScopes) {
      for (size_t i = 0; i < bound; ++i)
        targets.push_back(marker.chain[i]);
    } else if (marker.level >= 0) {
      TVM_FFI_CHECK(static_cast<size_t>(marker.level) < bound, ValueError)
          << "conflict-hint level " << marker.level << " is out of range (only "
          << bound << " common enclosing loop(s))";
      targets.push_back(marker.chain[marker.level]);
    }
    return targets;
  }

  void ResolveGroups() {
    std::unordered_map<std::string, size_t> pending;
    std::unordered_set<std::string> completed;
    for (size_t i = 0; i < markers_.size(); ++i) {
      Marker &marker = markers_[i];
      if (marker.group.empty()) {
        marker.hint_id = NewId("h", &next_hint_, &used_hints_);
        marker.targets = Targets(marker, marker.chain.size());
        continue;
      }
      const std::string key = (marker.polarity ? "conflict:" : "no-conflict:") +
                              std::string(marker.group);
      TVM_FFI_CHECK(!completed.count(key), ValueError)
          << "conflict-hint group '" << marker.group
          << "' appears more than twice; each group must appear exactly twice";
      auto previous = pending.find(key);
      if (previous == pending.end()) {
        marker.hint_id = NewId("h", &next_hint_, &used_hints_);
        marker.role = "lhs";
        marker.operands = {marker.operands[0]};
        pending.emplace(key, i);
      } else {
        Marker &first = markers_[previous->second];
        TVM_FFI_CHECK(first.level == marker.level, ValueError)
            << "conflict-hint group '" << marker.group
            << "' has mismatched level";
        TVM_FFI_CHECK(first.cross == marker.cross, ValueError)
            << "conflict-hint group '" << marker.group
            << "' has mismatched cross";
        auto targets = Targets(marker, CommonPrefix(first.chain, marker.chain));
        first.targets = targets;
        marker.targets = std::move(targets);
        marker.hint_id = first.hint_id;
        marker.role = "rhs";
        marker.operands = {marker.operands[0]};
        pending.erase(previous);
        completed.insert(key);
      }
    }
    if (!pending.empty())
      TVM_FFI_THROW(ValueError)
          << "conflict-hint group '" << markers_[pending.begin()->second].group
          << "' appears only once; each group must appear exactly twice";
  }

  String ScopeId(const ForObj *loop) {
    if (!loop)
      return "root";
    auto it = scope_ids_.find(loop);
    if (it != scope_ids_.end())
      return it->second;
    String id = NewId("l", &next_scope_, &used_scopes_);
    scope_ids_.emplace(loop, id);
    return id;
  }

  Fields RewriteMarker(const Marker &marker) {
    Fields output;
    Array<String> scoped_targets;
    for (const ForObj *target : marker.targets) {
      bool is_direct = marker.group.empty() &&
                       marker.direct_scope != kNoDirectScope &&
                       ((marker.direct_scope == kRoot && target == nullptr) ||
                        (marker.direct_scope >= 0 &&
                         target == marker.chain[marker.direct_scope]));
      if (is_direct) {
        Array<Any> entry{marker.operands[0], marker.operands[1],
                         IntImm::Int32(marker.cross),
                         IntImm(PrimType::Bool(), marker.polarity)};
        output.Set(target ? kDirect : kRootConflictHints, Array<Any>{entry});
      } else {
        scoped_targets.push_back(ScopeId(target));
      }
    }
    if (!scoped_targets.empty()) {
      Fields record{{"schema_version", 1},
                    {"hint_id", marker.hint_id},
                    {"role", marker.role},
                    {"operands", marker.operands},
                    {"target_scopes", scoped_targets},
                    {"cross", IntImm::Int32(marker.cross)},
                    {"polarity", IntImm(PrimType::Bool(), marker.polarity)}};
      output.Set(kScoped, Array<Any>{record});
    }
    return output;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    if (!IsCallNamed(op->value.as<CallObj>(), kMarker))
      return StmtExprMutator::Mutate_(op, mode);
    TVM_FFI_ICHECK_LT(cursor_, markers_.size());
    return MakeAnnotation(RewriteMarker(markers_[cursor_++]), op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    For loop = StmtExprMutator::Mutate_(op, mode)
                   .ValueOrUnchanged(GetRef<Stmt>(op))
                   .as_or_throw<For>();
    if (auto it = scope_ids_.find(op); it != scope_ids_.end()) {
      loop.CopyOnWrite()->annotations.Set(kScopeId, it->second);
    }
    return loop;
  }
};
} // namespace

tvm::transform::Pass NormalizeConflictHints() {
  auto pass = [](tirx::Function f, const IRModule &,
                 const tvm::transform::PassContext &) {
    return RewriteTilelangKernels(
        std::move(f), "NormalizeConflictHints",
        [](const TilelangKernelContext &context) {
          return ConflictHintNormalizer::Rewrite(context.root);
        },
        false);
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.NormalizeConflictHints");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.NormalizeConflictHints",
                              NormalizeConflictHints);
}
} // namespace tvm::tl
