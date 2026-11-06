/*!
 * \file simplify.cc
 * \brief Statement simplifier based on analyzer and remove useless parameters
 * of TL PrimFunc.
 */

#include "support/check.h"
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ir/cast.h>
#include <tvm/s_tir/utils.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "arith/const_fold.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "arith/product_normal_form.h"
#include "common/attr.h"
#include "tir/analysis/control_flow_graph.h"
#include "tir/analysis/var_use_def_analysis.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using namespace arith;

struct SimplifyConfigNode : public AttrsNodeReflAdapter<SimplifyConfigNode> {
  bool transitively_prove_inequalities{};
  bool propagate_knowns_to_prove_conditional{};
  bool propagate_knowns_to_simplify_expressions{};
  bool convert_boolean_to_and_of_ors{};
  bool apply_constraints_to_boolean_branches{};
  bool enable_simplify_let_inline{true};

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<SimplifyConfigNode>()
        .def_ro("transitively_prove_inequalities",
                &SimplifyConfigNode::transitively_prove_inequalities,
                "If true, simplify conditionals with transitive combinations "
                "of scoped constraints",
                refl::DefaultValue(false))
        .def_ro("propagate_knowns_to_prove_conditional",
                &SimplifyConfigNode::propagate_knowns_to_prove_conditional,
                "If true, known buffer values are propagated and used to "
                "statically prove conditionals",
                refl::DefaultValue(false))
        .def_ro("propagate_knowns_to_simplify_expressions",
                &SimplifyConfigNode::propagate_knowns_to_simplify_expressions,
                "If true, known buffer values are propagated and used to "
                "replace BufferLoad wherever "
                "possible",
                refl::DefaultValue(false))
        .def_ro("convert_boolean_to_and_of_ors",
                &SimplifyConfigNode::convert_boolean_to_and_of_ors,
                "If true, simplify conditionals into an AND of ORs",
                refl::DefaultValue(false))
        .def_ro("apply_constraints_to_boolean_branches",
                &SimplifyConfigNode::apply_constraints_to_boolean_branches,
                "If true, simplify each branch of AND/OR under a constraints "
                "provided by the other "
                "branch",
                refl::DefaultValue(false))
        .def_ro("enable_simplify_let_inline",
                &SimplifyConfigNode::enable_simplify_let_inline,
                "If true, inline let statements when possible",
                refl::DefaultValue(true));
  }
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.transform.SimplifyConfig",
                                    SimplifyConfigNode, BaseAttrsNode);

  RewriteSimplifier::Extension GetEnabledExtensions() const {
    RewriteSimplifier::Extension flags = RewriteSimplifier::kNone;
    if (transitively_prove_inequalities) {
      flags = RewriteSimplifier::Extension(
          flags | RewriteSimplifier::kTransitivelyProveInequalities);
    }
    if (convert_boolean_to_and_of_ors) {
      flags = RewriteSimplifier::Extension(
          flags | RewriteSimplifier::kConvertBooleanToAndOfOrs);
    }
    if (apply_constraints_to_boolean_branches) {
      flags = RewriteSimplifier::Extension(
          flags | RewriteSimplifier::kApplyConstraintsToBooleanBranches);
    }
    return flags;
  }
};

std::unordered_set<const BufferNode *>
CollectUsedBuffers(const PrimFunc &func) {
  struct Visitor : StmtExprVisitor {
    using StmtExprVisitor::VisitExpr_;
    using StmtExprVisitor::VisitStmt_;

    Visitor(PrimFunc func) : func(std::move(func)) {}

    void VisitExpr_(const CallNode *op) override {
      for (const auto &arg : op->args) {
        for (const auto &it : func->buffer_map) {
          if (Downcast<PrimExpr>(it.second.get()->data).same_as(arg)) {
            used_in_buffer_def_.insert(it.second.get());
          }
        }
      }
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitExpr_(const BufferLoadNode *op) override {
      VisitBuffer(op->buffer);
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) override {
      VisitBuffer(op->buffer);
      StmtExprVisitor::VisitStmt_(op);
    }
    void VisitStmt_(const SBlockNode *op) override {
      for (const auto &buffer : op->alloc_buffers) {
        for (const auto &it : func->buffer_map) {
          if (it.second.get()->data.same_as(buffer.get()->data)) {
            used_in_buffer_def_.insert(it.second.get());
          }
        }
      }
      for (const auto &buffer : op->reads) {
        for (const auto &it : func->buffer_map) {
          if (it.second.get()->data.same_as(buffer->buffer.get()->data)) {
            used_in_buffer_def_.insert(it.second.get());
          }
        }
      }
      for (const auto &buffer : op->writes) {
        for (const auto &it : func->buffer_map) {
          if (it.second.get()->data.same_as(buffer->buffer.get()->data)) {
            used_in_buffer_def_.insert(it.second.get());
          }
        }
      }
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitBuffer(const Buffer &buf) {
      // Collect buffers that should remain defined
      VarUseDefAnalyzer usage(Array<Var>{});
      usage(buf->data);
      for (const auto &dim : buf->shape) {
        usage(dim);
      }
      for (const auto &dim : buf->strides) {
        usage(dim);
      }
      usage(buf->elem_offset);

      for (const auto &buffer : usage.buffer_use_count_) {
        if (buffer.second >= 1) {
          used_in_buffer_def_.insert(buffer.first);
        }
      }
      for (const auto &buffer : usage.undefined_buffers_) {
        used_in_buffer_def_.insert(buffer.get());
      }
    }
    PrimFunc func;
    std::unordered_set<const BufferNode *> used_in_buffer_def_;
  };

  Visitor visitor(func);
  visitor(func->body);
  return visitor.used_in_buffer_def_;
}

/* \brief Collect variables referenced by buffer definitions. */
std::unordered_set<const VarNode *>
CollectVarsUsedInBufferDefinition(const Stmt &stmt) {
  struct Visitor : StmtExprVisitor {
    using StmtExprVisitor::VisitExpr_;
    using StmtExprVisitor::VisitStmt_;

    void VisitExpr_(const BufferLoadNode *op) override {
      VisitBuffer(op->buffer);
      StmtExprVisitor::VisitExpr_(op);
    }
    void VisitStmt_(const BufferStoreNode *op) override {
      VisitBuffer(op->buffer);
      StmtExprVisitor::VisitStmt_(op);
    }

    void VisitBuffer(const Buffer &buf) {
      // Collect variables that should remain defined
      VarUseDefAnalyzer usage(Array<Var>{});
      usage(buf->data);
      for (const auto &dim : buf->shape) {
        usage(dim);
      }
      for (const auto &dim : buf->strides) {
        usage(dim);
      }
      usage(buf->elem_offset);

      // Track for use in BindNode mutator
      for (const auto &var : usage.undefined_) {
        used_in_buffer_def_.insert(var.get());
      }
    }
    std::unordered_set<const VarNode *> used_in_buffer_def_;
  };

  Visitor visitor;
  visitor(stmt);
  return visitor.used_in_buffer_def_;
}

namespace {

struct BindUseInfo {
  using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
  VarSet used_vars;
  VarSet volatile_buffers;
};

class BindUseCollector : public StmtExprVisitor {
public:
  static BindUseInfo Collect(const PrimFunc &func) {
    BindUseCollector collector;
    collector.VisitObject_(func);
    return std::move(collector.info_);
  }

private:
  // IR children and metadata share the same deduplication entry point.
  void VisitStmt(const Stmt &stmt) final { VisitObject_(stmt); }
  void VisitExpr(const PrimExpr &expr) final { VisitObject_(expr); }

  void VisitExpr_(const VarNode *op) final {
    info_.used_vars.insert(GetRef<Var>(op));
    VisitObject_(op->type_annotation);
  }

  void VisitStmt_(const BindNode *op) final {
    // Skip the definition without marking its Var visited: metadata may use it
    // later.
    StmtExprVisitor::VisitStmt_(op);
    VisitObject_(op->var->type_annotation);
  }

  void VisitStmt_(const AllocBufferNode *op) final {
    if (op->annotations.count(tirx::attr::kVolatile)) {
      info_.volatile_buffers.insert(op->buffer->data);
    }
    StmtExprVisitor::VisitStmt_(op);
    VisitObject_(op->annotations);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::volatile_scope) {
      if (auto var = op->node.as<Var>()) {
        info_.volatile_buffers.insert(var.value());
      } else if (auto buffer = op->node.as<Buffer>()) {
        info_.volatile_buffers.insert(buffer.value()->data);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
    VisitObject_(op->node);
  }

  void VisitBufferDef(const Buffer &buffer, bool alloc_data) final {
    VisitObject_(buffer);
  }

  void VisitBufferUse(const Buffer &buffer) final {
    // Liveness must retain references in buffer descriptors even when there is
    // no explicit DeclBuffer in the body. It does not enforce lexical scope.
    VisitObject_(buffer);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    VisitObject_(op->predicate);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    VisitObject_(op->predicate);
  }

  void VisitStmt_(const ForNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    VisitObject_(op->loop_var);
    VisitObject_(op->thread_binding);
    VisitObject_(op->annotations);
  }

  void VisitStmt_(const SBlockNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    VisitObject_(op->iter_vars);
    VisitObject_(op->annotations);
  }

  void VisitExpr_(const CallNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    VisitObject_(op->op);
    VisitObject_(op->annotations);
  }

  void VisitExpr_(const CastNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    VisitObject_(op->annotations);
  }

  void VisitExpr_(const LetNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    VisitObject_(op->var);
  }

  void VisitExpr_(const ReduceNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    VisitObject_(op->combiner);
    VisitObject_(op->axis);
  }

  void VisitExpr_(const ProducerLoadNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    VisitObject_(op->producer);
  }

  void VisitExpr_(const RampNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    VisitExpr(op->lanes);
  }

  void VisitExpr_(const BroadcastNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    VisitExpr(op->lanes);
  }

  // Use ordinary IR visitors for statements and expressions. Reflection is
  // only needed for containers and metadata objects such as Buffer and Layout.
  void VisitObject_(const Any &value) {
    auto obj = value.as<ObjectRef>();
    if (!obj || !obj->defined() || !visited_.insert(*obj).second) {
      return;
    }
    if (auto expr = obj->as<PrimExpr>()) {
      StmtExprVisitor::VisitExpr(expr.value());
    } else if (auto stmt = obj->as<Stmt>()) {
      StmtExprVisitor::VisitStmt(stmt.value());
    } else if (auto array = obj->as<ArrayObj>()) {
      for (const Any &item : *array) {
        VisitObject_(item);
      }
    } else if (auto map = obj->as<MapObj>()) {
      for (const auto &item : *map) {
        VisitObject_(item.first);
        VisitObject_(item.second);
      }
    } else {
      auto visit_field = [&](const TVMFFIFieldInfo *field) {
        VisitObject_(reflection::FieldGetter(field)(*obj));
      };
      reflection::ForEachFieldInfo(TVMFFIGetTypeInfo((*obj)->type_index()),
                                   visit_field);
    }
  }

  BindUseInfo info_;
  std::unordered_set<ObjectRef, ObjectPtrHash, ObjectPtrEqual> visited_;
};

class UnusedBindRemover : public StmtMutator {
public:
  static PrimFunc Apply(PrimFunc func) {
    bool removed;
    // Recompute after removing consumers to handle chains and shared nodes.
    do {
      UnusedBindRemover remover;
      remover.uses_ = BindUseCollector::Collect(func);
      func.CopyOnWrite()->body = remover(func->body);
      removed = remover.removed_;
    } while (removed);
    return func;
  }

private:
  bool CanDiscard_(const PrimExpr &value) const {
    bool discard = SideEffect(value) <= CallEffectKind::kReadState &&
                   !UsesVar(value, [&](const VarNode *var) {
                     return uses_.volatile_buffers.count(GetRef<Var>(var));
                   });
    // SideEffect and UsesVar do not visit BufferLoad predicates.
    PostOrderVisit(value, [&](const ObjectRef &node) {
      if (const auto *load = node.as<BufferLoadNode>();
          load && load->predicate) {
        discard &= CanDiscard_(load->predicate.value());
      }
    });
    return discard;
  }

  Stmt VisitStmt_(const BindNode *op) final {
    if (!uses_.used_vars.count(op->var) && CanDiscard_(op->value)) {
      removed_ = true;
      return Evaluate(Integer(0));
    }
    return GetRef<Stmt>(op);
  }

  BindUseInfo uses_;
  bool removed_{false};
};

} // namespace

class SimplifyConfig : public Attrs {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(SimplifyConfig, Attrs,
                                                SimplifyConfigNode);
};
TVM_FFI_STATIC_INIT_BLOCK() { SimplifyConfigNode::RegisterReflection(); }

TVM_REGISTER_PASS_CONFIG_OPTION("tl.Simplify", SimplifyConfig);

// Remainder canonicalization and factor matching substantially follow the
// approach in penguin-wwy's PR #3267. Match reassociated products without
// changing arithmetic widths or distributing through casts.
class LayoutDivModSimplifier : public StmtExprMutator {
public:
  explicit LayoutDivModSimplifier(arith::Analyzer *analyzer)
      : analyzer_(analyzer) {}

  PrimExpr Simplify(const PrimExpr &expr) { return VisitExpr(expr); }

  PrimExpr VisitExpr_(const FloorModNode *op) final {
    PrimExpr value = StmtExprMutator::VisitExpr_(op);
    const auto *mod = value.as<FloorModNode>();
    const auto *radix = mod ? mod->b.as<IntImmNode>() : nullptr;
    if (radix && radix->value > 0 && (radix->value & (radix->value - 1)) == 0 &&
        IsSupportedInteger(value.dtype()) &&
        SideEffect(value) == CallEffectKind::kPure) {
      if (auto reduced = RecoverRemainder_(mod->a, mod->b)) {
        return floormod(reduced.value(), mod->b);
      }
    }
    return value;
  }

  PrimExpr VisitExpr_(const SubNode *op) final {
    PrimExpr value = StmtExprMutator::VisitExpr_(op);
    const auto *sub = value.as<SubNode>();
    if (!sub || !IsSupportedInteger(value.dtype()) ||
        SideEffect(value) != CallEffectKind::kPure) {
      return value;
    }
    if (auto divisor = MatchProduct_(sub->b, sub->a)) {
      return floormod(sub->a, divisor.value());
    }
    if (auto reduced = RecoverRemainder_(value)) {
      return reduced.value();
    }
    return value;
  }

  PrimExpr VisitExpr_(const AddNode *op) final {
    PrimExpr value = StmtExprMutator::VisitExpr_(op);
    const auto *add = value.as<AddNode>();
    if (!add || SideEffect(value) != CallEffectKind::kPure) {
      return value;
    }
    if (IsSupportedInteger(value.dtype())) {
      for (int i = 0; i < 2; ++i) {
        PrimExpr product = i == 0 ? add->a : add->b;
        PrimExpr remainder = i == 0 ? add->b : add->a;
        if (const auto *mod = remainder.as<FloorModNode>()) {
          auto divisor = MatchProduct_(product, mod->a);
          if (divisor && ffi::StructuralEqual()(divisor.value(), mod->b)) {
            // q*d + r = x also holds under signed word wrapping.
            value = mod->a;
            break;
          }
        }
      }
    }
    const auto *sum = value.as<AddNode>();
    if (!sum) {
      return value;
    }
    PrimExpr left = sum->a;
    PrimExpr right = sum->b;
    if (auto result = Combine_(left, right)) {
      return result.value();
    }
    if (auto result = Combine_(right, left)) {
      return result.value();
    }
    return value;
  }

private:
  arith::Analyzer *analyzer_;

  bool IsSupportedInteger(DataType dtype) {
    return dtype.is_scalar() && (dtype.is_int() || dtype.is_uint()) &&
           (dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32 ||
            dtype.bits() == 64);
  }

  // Drop only casts that preserve every value of the source type. In
  // particular, sign extension into an unsigned type is not a value-preserving
  // widening.
  bool IsValuePreservingWiden(DataType from, DataType to) {
    return IsSupportedInteger(from) && IsSupportedInteger(to) &&
           from.bits() < to.bits() && (from.is_uint() || to.is_int());
  }

  bool FitsInType(const arith::ConstIntBound &bound, DataType dtype) {
    int bits = dtype.bits();
    int64_t minimum = dtype.is_uint()
                          ? 0
                          : (bits == 64 ? std::numeric_limits<int64_t>::min()
                                        : -(int64_t{1} << (bits - 1)));
    uint64_t maximum =
        dtype.is_int() ? (uint64_t{1} << (bits - 1)) - 1
                       : (std::numeric_limits<uint64_t>::max() >> (64 - bits));
    return bound->min_value != arith::ConstIntBoundNode::kNegInf &&
           bound->max_value != arith::ConstIntBoundNode::kPosInf &&
           bound->min_value >= minimum &&
           (bound->max_value < 0 || uint64_t(bound->max_value) <= maximum);
  }

  // Analyzer and its Z3 prover use mathematical integers. Only ask them for
  // word properties when each intermediate and cast agrees with that
  // interpretation. Bound-preserving casts are legal even when the source type
  // itself is wider.
  bool HasUnprovenWrap(const PrimExpr &expr, arith::Analyzer *analyzer) {
    bool may_wrap = false;
    PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
      if (may_wrap) {
        return;
      }
      if (const auto *cast = node.as<CastNode>()) {
        may_wrap =
            !IsSupportedInteger(cast->value.dtype()) ||
            !IsSupportedInteger(cast->dtype) ||
            (cast->dtype != cast->value.dtype() &&
             !IsValuePreservingWiden(cast->value.dtype(), cast->dtype) &&
             !FitsInType(analyzer->const_int_bound(cast->value), cast->dtype));
      } else if (const auto *mul = node.as<MulNode>()) {
        may_wrap = !IsSupportedInteger(mul->dtype) ||
                   !ProductFits(mul->a, mul->b, analyzer, false);
      } else if (node.as<AddNode>() || node.as<SubNode>()) {
        // These composite bounds may describe an already-wrapped result.
        // Leave them unchanged rather than build a second range analyzer.
        may_wrap = true;
      }
    });
    return may_wrap;
  }

  bool ProductFits(const PrimExpr &a, const PrimExpr &b,
                   arith::Analyzer *analyzer, bool positive) {
    auto left = analyzer->const_int_bound(a);
    auto right = analyzer->const_int_bound(b);
    int64_t minimum = positive ? 1 : 0;
    if (left->min_value < minimum || right->min_value < minimum ||
        left->max_value == arith::ConstIntBoundNode::kPosInf ||
        right->max_value == arith::ConstIntBoundNode::kPosInf) {
      return false;
    }
    DataType dtype = a.dtype();
    uint64_t maximum = dtype.is_int() ? (uint64_t{1} << (dtype.bits() - 1)) - 1
                                      : std::numeric_limits<uint64_t>::max() >>
                                            (64 - dtype.bits());
    // A constraint on the wrapped product cannot prove that it did not wrap.
    // Divide the type limit instead of multiplying the operand upper bounds.
    return right->max_value == 0 ||
           uint64_t(left->max_value) <= maximum / uint64_t(right->max_value);
  }

  ffi::Optional<PrimExpr> Combine_(const PrimExpr &product,
                                   const PrimExpr &remainder) {
    const auto *mul = product.as<MulNode>();
    const auto *mod = remainder.as<FloorModNode>();
    if (!mul || !mod) {
      return std::nullopt;
    }
    const auto *inner = mod->a.as<FloorDivNode>();
    if (!inner || !IsSupportedInteger(inner->dtype) ||
        HasUnprovenWrap(inner->b, analyzer_) ||
        HasUnprovenWrap(mod->b, analyzer_) ||
        !ProductFits(inner->b, mod->b, analyzer_, true)) {
      return std::nullopt;
    }
    for (int i = 0; i < 2; ++i) {
      PrimExpr quotient = i == 0 ? mul->a : mul->b;
      PrimExpr radix = i == 0 ? mul->b : mul->a;
      const auto *outer = quotient.as<FloorDivNode>();
      if (!outer || !ffi::StructuralEqual()(radix, mod->b) ||
          !ffi::StructuralEqual()(outer->a, inner->a)) {
        continue;
      }
      PrimExpr combined = inner->b * mod->b;
      if (SameProduct_(outer->b, combined)) {
        // q = floor(x/a); floor(x/(a*b))*b + q%b = q. Positive
        // denominators and non-wrapping products justify the factorization.
        return mod->a;
      }
    }
    return std::nullopt;
  }

  // Simplification can interleave a swizzle's offsets with x - (x/d)*d.
  // Recover the remainder without distributing products or crossing casts.
  ffi::Optional<PrimExpr>
  RecoverRemainder_(const PrimExpr &value,
                    ffi::Optional<PrimExpr> modulus = std::nullopt) {
    std::vector<std::pair<PrimExpr, bool>> terms;
    arith::UnpackSum(value, [&](const PrimExpr &term, int sign) {
      terms.emplace_back(term, sign < 0);
    });
    if (terms.size() > 32) {
      return std::nullopt;
    }
    for (size_t i = 0; i < terms.size(); ++i) {
      if (!terms[i].second) {
        continue;
      }
      std::vector<PrimExpr> factors;
      CollectFactors_(terms[i].first, &factors);
      for (const PrimExpr &factor : factors) {
        const auto *div = factor.as<FloorDivNode>();
        if (!div || !MatchProduct_(terms[i].first, div->a)) {
          continue;
        }
        std::vector<std::pair<PrimExpr, bool>> dividend;
        arith::UnpackSum(div->a, [&](const PrimExpr &term, int sign) {
          dividend.emplace_back(term, sign < 0);
        });
        std::vector<bool> used(terms.size(), false);
        used[i] = true;
        bool matched = true;
        for (const auto &term : dividend) {
          size_t j = 0;
          for (; j < terms.size(); ++j) {
            if (!used[j] && term.second == terms[j].second &&
                ffi::StructuralEqual()(term.first, terms[j].first)) {
              used[j] = true;
              break;
            }
          }
          if (j == terms.size()) {
            // A power-of-two outer modulus also discards word overflow.
            // Missing multiples of it can be restored before matching x-q*d.
            if (modulus.defined() && !HasUnprovenWrap(term.first, analyzer_) &&
                analyzer_->CanProve(floormod(term.first, modulus.value()) ==
                                    0)) {
              continue;
            }
            matched = false;
            break;
          }
        }
        if (matched) {
          PrimExpr result = floormod(div->a, div->b);
          for (size_t j = 0; j < terms.size(); ++j) {
            if (!used[j]) {
              result = terms[j].second ? result - terms[j].first
                                       : result + terms[j].first;
            }
          }
          return result;
        }
      }
    }
    return std::nullopt;
  }

  static ffi::Optional<PrimExpr> MatchProduct_(const PrimExpr &product,
                                               const PrimExpr &dividend) {
    std::vector<PrimExpr> factors;
    CollectFactors_(product, &factors);
    for (size_t i = 0; i < factors.size(); ++i) {
      const auto *quotient = factors[i].as<FloorDivNode>();
      if (!quotient || !ffi::StructuralEqual()(quotient->a, dividend)) {
        continue;
      }
      std::vector<PrimExpr> divisor_factors;
      CollectFactors_(quotient->b, &divisor_factors);
      auto remaining = factors;
      remaining.erase(remaining.begin() + i);
      if (SameFactors_(remaining, divisor_factors)) {
        return quotient->b;
      }
    }
    return std::nullopt;
  }

  static bool SameProduct_(const PrimExpr &a, const PrimExpr &b) {
    if (a.dtype() != b.dtype())
      return false;
    std::vector<PrimExpr> left, right;
    CollectFactors_(a, &left);
    CollectFactors_(b, &right);
    return SameFactors_(left, right);
  }

  static bool SameFactors_(const std::vector<PrimExpr> &left,
                           const std::vector<PrimExpr> &right) {
    return std::is_permutation(left.begin(), left.end(), right.begin(),
                               right.end(), ffi::StructuralEqual());
  }

  static void CollectFactors_(const PrimExpr &value,
                              std::vector<PrimExpr> *factors) {
    arith::UnpackReduction<MulNode>(
        value, [&](const PrimExpr &factor) { factors->push_back(factor); });
  }
};

class StmtSimplifier : public IRMutatorWithAnalyzer {
public:
  static PrimFunc
  Apply(PrimFunc func, Analyzer *analyzer,
        const Optional<SimplifyConfig> &config_opt = std::nullopt,
        bool simplify_arguments = false) {
    auto config = config_opt.value_or(AttrsWithDefaultValues<SimplifyConfig>());
    analyzer->rewrite_simplify.SetEnabledExtensions(
        config->GetEnabledExtensions());

    std::optional<ControlFlowGraph> touch_pattern = std::nullopt;
    if (config->propagate_knowns_to_prove_conditional ||
        config->propagate_knowns_to_simplify_expressions) {
      touch_pattern = ControlFlowGraph(func->body);
    }

    std::unordered_set<const VarNode *> used_in_buffer_def =
        CollectVarsUsedInBufferDefinition(func->body);
    StmtSimplifier simplifier(analyzer, config, std::move(touch_pattern),
                              std::move(used_in_buffer_def));
    simplifier.MarkBufferMapShapes(func);
    func.CopyOnWrite()->body = simplifier(func->body);
    func = UnusedBindRemover::Apply(std::move(func));

    // Optionally remove unused buffer parameters
    if (simplify_arguments) {
      // First get used buffers
      simplifier.used_buffers_ = CollectUsedBuffers(func);

      bool param_updated = false;
      Array<Var> new_params;
      Map<Var, Buffer> new_buffer_map;
      // Check whether each buffer is used
      for (const auto &var : func->params) {
        if (func->buffer_map.find(var) != func->buffer_map.end()) {
          if (simplifier.used_buffers_.find(func->buffer_map[var].get()) !=
              simplifier.used_buffers_.end()) {
            new_params.push_back(var);
            new_buffer_map.Set(var, func->buffer_map[var]);
          } else if (simplifier.used_in_buffer_def_.find(
                         func->buffer_map[var]->data.get()) !=
                     simplifier.used_in_buffer_def_.end()) {
            new_params.push_back(var);
            new_buffer_map.Set(var, func->buffer_map[var]);
          } else {
            param_updated = true;
          }
        } else {
          // Non-buffer parameters (e.g., scalars) are always retained
          new_params.push_back(var);
        }
      }

      if (param_updated) {
        return PrimFunc(new_params, func.CopyOnWrite()->body, func->ret_type,
                        new_buffer_map, func->attrs, func->span);
      }
    }
    // Either no change to params or argument simplification disabled
    return func;
  }

private:
  // TileLang-local post-simplification: TVM's rewrite simplifier can prove
  // singleton bounds for floor-div expressions, but does not always rewrite the
  // expression to that singleton constant.  Keep this scoped to tl.Simplify so
  // internal lowering passes are not affected by a global arithmetic rewrite.
  class ContextSingletonFloorDivFolder : public StmtExprMutator {
  public:
    explicit ContextSingletonFloorDivFolder(Analyzer *analyzer)
        : analyzer_(analyzer) {}

    PrimExpr Fold(const PrimExpr &expr) { return VisitExpr(expr); }

  private:
    using StmtExprMutator::VisitExpr_;

    PrimExpr VisitExpr_(const FloorDivNode *op) final {
      PrimExpr expr = StmtExprMutator::VisitExpr_(op);
      expr = analyzer_->Simplify(expr);
      if (const auto *div = expr.as<FloorDivNode>();
          div && IsIndexType(div->dtype)) {
        // Example: under thread_extent tx in [0, 256) and else(tx < 128),
        // const_int_bound(tx // 128) is [1, 1], so the div is constant.
        ConstIntBound bound = analyzer_->const_int_bound(expr);
        if (bound.defined() && bound->min_value == bound->max_value) {
          return make_const(div->dtype, bound->min_value);
        }
      }
      return expr;
    }

    Analyzer *analyzer_;
  };

  explicit StmtSimplifier(
      Analyzer *analyzer, SimplifyConfig config,
      std::optional<ControlFlowGraph> touch_pattern,
      std::unordered_set<const VarNode *> used_in_buffer_def)
      : IRMutatorWithAnalyzer(analyzer), config_(std::move(config)),
        touch_pattern_(std::move(touch_pattern)),
        used_in_buffer_def_(std::move(used_in_buffer_def)) {}

  using Parent = IRMutatorWithAnalyzer;
  using Parent::VisitExpr_;
  using Parent::VisitStmt;
  using Parent::VisitStmt_;

  PrimExpr VisitExpr(const PrimExpr &expr) final {
    PrimExpr simplified;
    if (config_->propagate_knowns_to_simplify_expressions) {
      simplified = touch_pattern_->SimplifyInContext(
          expr, current_stmt_.value(), analyzer_);
    } else {
      simplified = analyzer_->Simplify(expr);
    }

    simplified = LayoutDivModSimplifier(analyzer_).Simplify(simplified);
    ContextSingletonFloorDivFolder folder(analyzer_);
    PrimExpr folded = folder.Fold(simplified);
    if (!folded.same_as(simplified)) {
      // Clean up arithmetic exposed by singleton folding, e.g.
      // 1 * 64 + i - 64 -> i.
      return analyzer_->Simplify(folded);
    }
    return simplified;
  }

  Stmt Simplify(Stmt stmt) { return operator()(std::move(stmt)); }

  Stmt VisitStmt(const Stmt &stmt) override {
    Optional<Stmt> cache = this->current_stmt_;
    this->current_stmt_ = stmt;
    Stmt output = Parent::VisitStmt(stmt);
    this->current_stmt_ = std::move(cache);
    return output;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (analyzer_->CanProve(op->extent <= 0)) {
      // Remove loops with non-positive extent
      return Evaluate(0);
    }
    With<ConstraintContext> ctx1(analyzer_, op->loop_var >= op->min);
    With<ConstraintContext> ctx2(analyzer_,
                                 op->loop_var < op->min + op->extent);
    return Parent::VisitStmt_(op);
  }

  bool CanInlineBind(const BindNode *op) {
    if (!config_->enable_simplify_let_inline)
      return false;
    if (is_const_number(op->value))
      return true;
    if (op->value.as<VarNode>())
      return true;
    // Won't face the deep expression explosion problem as in Let expression.
    // attempt to inline as much as possible if the value integer type(can be
    // index).
    if (!op->value.dtype().is_int())
      return false;
    return SideEffect(op->value) <= CallEffectKind::kPure;
  }

  Stmt VisitStmt_(const BindNode *op) override {
    PrimExpr value = this->VisitExpr(op->value);
    bool remove_buffer_alias = false;
    // TileLang emits aliases like `X_shared = buffer[0:128, 0:32]` to annotate
    // fragment types. TVM currently reinterprets vectorized/shared accesses as
    // Let-bound BufferLoad/BufferRegion nodes. If these bindings survive, later
    // passes (Layout rewrite, FlattenBuffer) substitute them with vector lanes
    // that our layout can't handle. Force-inline (by dropping the let) whenever
    // the alias spans more than 2 dims or carries vector lanes.
    auto get_ranges = [&](const PrimExpr &expr) -> Array<Range> {
      Array<Range> ranges;
      if (const auto *load = expr.as<BufferLoadNode>()) {
        for (const PrimExpr &index : load->indices) {
          if (const auto *ramp = index.as<RampNode>()) {
            ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
          } else {
            ranges.push_back(Range::FromMinExtent(index, Integer(1)));
          }
        }
      } else if (const auto *region = expr.as<BufferRegionNode>()) {
        for (const Range &range : region->region) {
          ranges.push_back(range);
        }
      }
      return ranges;
    };
    Array<Range> ranges = get_ranges(value);
    if (!ranges.empty()) {
      int non_unit_dims = 0;
      for (const Range &range : ranges) {
        PrimExpr extent = analyzer_->Simplify(range->extent);
        if (is_const_int(extent, 1) || analyzer_->CanProveEqual(extent, 1)) {
          continue;
        }
        ++non_unit_dims;
        if (non_unit_dims > 1) {
          remove_buffer_alias = true;
          break;
        }
      }
    }
    if (remove_buffer_alias) {
      return Evaluate(Integer(0));
    }

    bool can_inline = CanInlineBind(op);
    if (can_inline) {
      analyzer_->Bind(op->var, value);
    } else if (SideEffect(op->value) <= CallEffectKind::kPure) {
      non_inlined_bindings_.Set(op->var, value);
    }

    // Defer deletion to UnusedBindRemover: annotations and other metadata may
    // still reference the variable after its ordinary uses have been inlined.
    if (value.same_as(op->value)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->value = std::move(value);
      return Stmt(n);
    }
  }

  Stmt VisitStmt_(const IfThenElseNode *op) override {
    if (Optional<Bool> cond = ProveCondition(op->condition)) {
      if (cond.value()->value) {
        return this->VisitStmt(op->then_case);
      } else if (op->else_case) {
        return this->VisitStmt(op->else_case.value());
      } else {
        return Evaluate(0);
      }
    } else {
      return Parent::VisitStmt_(op);
    }
  }

  PrimExpr VisitExpr_(const CallNode *op) override {
    if (op->op.same_as(builtin::if_then_else())) {
      if (Optional<Bool> cond = ProveCondition(op->args[0])) {
        if (cond.value()->value) {
          return this->VisitExpr(op->args[1]);
        } else {
          return this->VisitExpr(op->args[2]);
        }
      }
    }
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const VarNode *op) override {
    used_vars_.insert(op);
    return Parent::VisitExpr_(op);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) override {
    auto buffer = op->buffer.get();
    if (used_buffers_.find(buffer) == used_buffers_.end()) {
      used_buffers_.insert(buffer);
    }
    return Parent::VisitExpr_(op);
  }

  // eliminate useless stores
  Stmt VisitStmt_(const BufferStoreNode *op) override {
    BufferStore store = Downcast<BufferStore>(Parent::VisitStmt_(op));
    if (const BufferLoadNode *load = store->value.as<BufferLoadNode>()) {
      if (load->buffer->data.same_as(store->buffer->data) &&
          ArrayDeepEqual(load->indices, store->indices) &&
          tirx::ExprDeepEqual()(load->buffer->elem_offset,
                                store->buffer->elem_offset) &&
          ArrayDeepEqual(load->buffer->shape, store->buffer->shape) &&
          ArrayDeepEqual(load->buffer->strides, store->buffer->strides)) {
        return Evaluate(0);
      }
    }
    auto buffer = op->buffer.get();
    if (used_buffers_.find(buffer) == used_buffers_.end()) {
      used_buffers_.insert(buffer);
    }
    return std::move(store);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) override {
    if (op->attr_key == "tl.assume") {
      PrimExpr condition = this->VisitExpr(Downcast<PrimExpr>(op->node));
      auto n = CopyOnWrite(op);
      n->node = std::move(condition);
      return Parent::VisitStmt_(n.get());
    }
    return Parent::VisitStmt_(op);
  }

private:
  bool ArrayDeepEqual(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs) {
    if (lhs.size() != rhs.size()) {
      return false;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
      if (!tirx::ExprDeepEqual()(lhs[i], rhs[i])) {
        return false;
      }
    }
    return true;
  }

  /* \brief Internal utility for checking conditionals
   *
   * Uses more aggressive optimization, such as performing additional
   * inlining and tracking known buffer values.
   */
  Optional<Bool> ProveCondition(PrimExpr condition) const {
    condition = Substitute(condition, non_inlined_bindings_);
    if (config_->propagate_knowns_to_prove_conditional) {
      ICHECK(touch_pattern_.has_value());
      condition = touch_pattern_->SimplifyInContext(
          condition, current_stmt_.value(), analyzer_);
    } else {
      condition = analyzer_->Simplify(condition);
    }
    if (const int64_t *as_int = as_const_int(condition)) {
      return Bool(*as_int);
    } else {
      // May have symbolic, need kSymbolicBound level prover.
      if (analyzer_->CanProve(condition) ||
          analyzer_->CanProve(condition,
                              arith::ProofStrength::kSymbolicBound)) {
        return Bool(true);
      }
      return std::nullopt;
    }
  }

  SimplifyConfig config_;
  std::optional<ControlFlowGraph> touch_pattern_;

  Map<Var, PrimExpr> non_inlined_bindings_;
  Optional<Stmt> current_stmt_{std::nullopt};
  std::unordered_set<const VarNode *> used_in_buffer_def_;
  std::unordered_set<const VarNode *> used_vars_;
  std::unordered_set<const BufferNode *> used_buffers_;
};

using namespace tirx::transform;

tvm::transform::Pass Simplify(bool simplify_arguments = true) {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    arith::Analyzer analyzer;
    auto cfg = ctx->GetConfig<SimplifyConfig>("tl.Simplify");
    return StmtSimplifier::Apply(std::move(f), &analyzer, cfg,
                                 simplify_arguments);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.Simplify", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.Simplify", Simplify);
}

} // namespace tl
} // namespace tvm
