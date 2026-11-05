/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file lower_opaque_block.cc
 * \brief Finish consumed TileLang metadata without dropping executable
 * contracts.
 */

#include "../ir/annotations.h"
#include "../op/native_utils.h"
#include "ir/lexical_scope.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <unordered_set>

namespace tvm::tl {
namespace {
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

Map<String, Any> DecodeMetadata(const CallObj *call) {
  auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
  const auto decoded =
      DecodeAnnotations(call->args[0].as_or_throw<tvm::Tuple>(),
                        attrs.GetAttr<Array<Any>>("annotation_schema").value())
          .cast<Map<Any, Any>>();
  Map<String, Any> result;
  for (const auto &[key, value] : decoded)
    result.Set(key.cast<String>(), value);
  return result;
}

Stmt Attribute(const Any &node, const String &key, const Any &value,
               SeqStmt body, Location loc) {
  auto encoded = EncodeAnnotations(
      Map<String, Any>{{"node", node}, {"key", key}, {"value", value}});
  return RegionStmt(
      Op::Get("tl.attr"), {encoded.payload}, {},
      DictAttrs({{"schema_version", 1}, {"annotation_schema", encoded.schema}}),
      body, {}, loc);
}

// Removing metadata does not remove the definition scope it surrounded. A
// SeqStmt constructor flattens nested sequences; a native scope is necessary
// for allocations and binder-aware transforms even when code executes once.
Stmt Lexical(SeqStmt body, Location loc, bool explicit_alloc_scope = false) {
  return MakeLexicalScope(std::move(body), std::move(loc),
                          explicit_alloc_scope);
}

class AnnotationFinalizer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  AnnotationFinalizer(const tirx::Function &function,
                      const Array<String> &allowed_pending_regions)
      : tile_ops_lowered_(
            function->GetAttr<bool>("tl.tile_ops_lowered").value_or(false)),
        safety_disabled_(
            tvm::transform::PassContext::Current()
                ->GetConfig<bool>("tl.disable_safe_memory_legalize")
                .value_or(false)) {
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    is_ascend_ = target && target.value()->kind->name == "ascend";
    if (target)
      target_kind_ = target.value()->kind->name;
    for (const auto &key : allowed_pending_regions)
      allowed_pending_regions_.insert(key);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    // An annotation statement describes the statements following it in this
    // lexical sequence. Convert code-generation annotations into explicit
    // regions over that suffix, keeping all dynamic operands in the IR.
    SeqStmt suffix(Array<Stmt>{}, op->loc);
    for (size_t i = op->seq.size(); i-- > 0;) {
      const Stmt &child = op->seq[i];
      const auto *evaluate = child.as<EvaluateObj>();
      const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
      if (call && call->op.same_as(Op::Get("tl.annotation"))) {
        suffix = Apply(DecodeMetadata(call), suffix, child->loc);
      } else {
        Stmt rewritten = Mutate(child, mode).ValueOrUnchanged(child);
        suffix = SeqStmt({rewritten, suffix}, op->loc);
      }
    }
    return suffix;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (op->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(op, mode);
    const auto fields = DecodeAttributeRegion(op);
    const String key = fields.at("key").cast<String>();
    if (key == "tl.size_var_defaults") {
      // The packed binder consumes default domains after annotation lowering.
      // Use native traversal so a definition remap also reaches its operands.
      return StmtExprMutator::Mutate_(op, mode);
    }
    SeqStmt body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    if (key == "lexical_alloc_scope")
      return Lexical(body, op->loc, true);
    if (key == "tl.sblock.init") {
      // BlockFrame/WorkloadBuilder have already emitted the initializer's
      // first-reduction-coordinate condition as an ordinary enclosing If.
      // Removing this descriptive wrapper must not hoist its body across that
      // If or flatten allocations/bindings into the surrounding block.
      return Lexical(body, op->loc);
    }
    if (key == "tl.sblock") {
      auto block = ToStringMap(fields.at("value"));
      String block_name;
      if (auto name = block.Get("name")) {
        block_name = name->as<StringImmObj>() ? name->as<StringImmObj>()->value
                                              : name->cast<String>();
      }
      const bool vf_region = block_name == "SIMT_VF" || block_name == "SIMD_VF";
      const bool execution_region =
          block_name == "CUBE" || block_name == "VECTOR" || vf_region;
      TVM_FFI_CHECK(!execution_region || is_ascend_, ValueError)
          << "Ascend execution region " << block_name
          << " requires an Ascend target";
      Map<String, Any> annotations;
      if (auto values = block.Get("annotations"))
        annotations = ToStringMap(*values);
      for (const auto &[name, value] : block) {
        TVM_FFI_CHECK(name == "name" || name == "no_realize" ||
                          name == "annotations",
                      ValueError)
            << "Unconsumed structured block field `" << name << "`";
      }
      for (const auto &[name, value] : annotations) {
        if (name == "lexical_alloc_scope")
          continue;
        const bool vector_count =
            block_name == "VECTOR" && name == "vector_count";
        const bool source_index = vf_region && name == "tl.vf_source_index";
        TVM_FFI_CHECK(vector_count || source_index, ValueError)
            << "Unconsumed structured block annotation `" << name << "`";
        const auto *literal = value.as<IntImmObj>();
        TVM_FFI_CHECK(literal || value.try_cast<int64_t>().has_value(),
                      ValueError)
            << "Ascend execution-region annotation " << name
            << " must be a constant integer";
        const int64_t count = literal ? static_cast<int64_t>(literal->value)
                                      : value.cast<int64_t>();
        if (vector_count) {
          TVM_FFI_CHECK(count == 1 || count == 2, ValueError)
              << "VECTOR vector_count must be a constant integer 1 or 2";
        } else {
          TVM_FFI_CHECK_GE(count, 0, ValueError)
              << "VF source index must be nonnegative";
        }
      }
      // CUBE/VECTOR select execution cores; SIMD_VF/SIMT_VF select the helper
      // calling convention and instruction domain in the Ascend/PTO emitters.
      // Keep their native operands and definition scope through finalization.
      if (execution_region) {
        return RegionStmt(op->op, op->args, op->body_params, op->attrs, body,
                          op->result_vars, op->loc);
      }
      // Ordinary BlockFrame axes and predicates are already native Binds/Ifs.
      return Lexical(body, op->loc, annotations.count("lexical_alloc_scope"));
    }
    if (key == "tl.kernel") {
      return Lexical(Apply(ToStringMap(fields.at("value")), body, op->loc),
                     op->loc);
    }
    if (key == "tl.simtvf_scope" || key == "tl.simdvf_scope") {
      TVM_FFI_CHECK(is_ascend_, ValueError)
          << "Ascend VF scope " << key << " requires an Ascend target";
      // PTO consumes SIMD scopes as vecscope/memory barriers. Both emitters
      // retain the surrounding VF block to select their execution domain.
      return RegionStmt(op->op, op->args, op->body_params, op->attrs, body,
                        op->result_vars, op->loc);
    }
    if (IsRetired(key))
      return Lexical(body, op->loc);
    if (key == "tl.loop.annotations") {
      Map<String, Any> values = ToStringMap(fields.at("value"));
      for (const auto &[name, unused] : values) {
        TVM_FFI_CHECK(IsRetiredLoopHint(name), ValueError)
            << "TileLang loop annotation `" << name
            << "` still needs its owning lowering pass";
      }
      return Lexical(body, op->loc);
    }
    // These regions have downstream consumers (codegen, CPU parallelism, or
    // per-function alias analysis). Early CUDA finalization may also retain
    // the two WS scopes until shared-memory planning, synchronization and
    // register allocation have consumed them. Preserve every native operand
    // and definition scope; a later strict finalizer checks consumption.
    if (IsCodegenKey(key) || key == "tl.non_restrict_params" ||
        key == "tl.cpu_num_threads" ||
        key == "tl.assume_requires_runtime_check" ||
        allowed_pending_regions_.count(key)) {
      return RegionStmt(op->op, op->args, op->body_params, op->attrs, body,
                        op->result_vars, op->loc);
    }
    TVM_FFI_CHECK(!RequiresOwningPass(key), ValueError)
        << "Unconsumed TileLang region `" << key
        << "` at FinalizeAnnotations; run its owning lowering pass first";
    // Generic user hints have no executable consumer. The old AttrStmt
    // emitters ignored their metadata and executed the body once. Keep the
    // native definition scope instead of flattening its Binds/allocations.
    return Lexical(body, op->loc);
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    if (const auto *intrinsic = op->op.as<OpObj>()) {
      TVM_FFI_CHECK(intrinsic->name.find("tl.tileop.") != 0, ValueError)
          << "FinalizeAnnotations cannot run before LowerTileOp: "
          << intrinsic->name;
    }
    // Source alloc_buffer's origin affects CPU lifetime placement only. Other
    // targets keep their native lexical allocation and consume this marker
    // before storage planning. A missing CPU preparation step cannot be
    // repaired by discarding the marker after block scopes were finalized.
    if (op->op.same_as(tirx::alloc_tensor_op()) && op->attrs) {
      auto attributes = op->attrs.value().as<DictAttrs>();
      if (attributes &&
          attributes.value()->dict.count("tl.source_block_alloc")) {
        const auto origin =
            attributes.value()->dict.at("tl.source_block_alloc").as<IntImm>();
        TVM_FFI_CHECK(origin && origin.value()->value == 1, ValueError)
            << "tl.source_block_alloc must be the source allocation marker";
        TVM_FFI_CHECK(!target_kind_.empty(), ValueError)
            << "Finalizing source allocation origins requires a bound target";
        TVM_FFI_CHECK(target_kind_ != "c" && target_kind_ != "llvm", ValueError)
            << "CPU source allocation origins require LowerCPUKernelLaunch "
               "before FinalizeAnnotations";
        auto rewritten = StmtExprMutator::Mutate_(op, mode)
                             .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                             .as_or_throw<Call>();
        auto values = rewritten->attrs.value().as_or_throw<DictAttrs>()->dict;
        values.erase("tl.source_block_alloc");
        return Call(rewritten->ty, rewritten->op, rewritten->args,
                    DictAttrs(values), rewritten->ty_args, rewritten->loc);
      }
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

private:
  static Map<String, Any> ToStringMap(const Any &input) {
    Map<String, Any> result;
    for (const auto &[key, value] : input.cast<Map<Any, Any>>())
      result.Set(key.cast<String>(), value);
    return result;
  }

  bool IsRetired(const String &key) const {
    if (key == "layout_map") {
      TVM_FFI_CHECK(tile_ops_lowered_, ValueError)
          << "layout_map cannot be discarded before LowerTileOp has "
             "materialized tensor layouts";
      return true;
    }
    if (key == "safe_value_map") {
      TVM_FFI_CHECK(safety_disabled_, ValueError)
          << "safe_value_map must be consumed by LegalizeSafeMemoryAccess";
      return true;
    }
    return false;
  }

  bool IsRetiredLoopHint(const String &key) const {
    static const std::unordered_set<std::string> hints{
        "prefer_async",         "async_scope",     "disable_tma",
        "disable_async",        "coalesced_width", "vectorize",
        "parallel_loop_layout", "predicate",       "padding_guard",
        "loop_layout",          "tl.ws_op_id"};
    if (!hints.count(key))
      return false;
    TVM_FFI_CHECK(tile_ops_lowered_, ValueError)
        << "Loop lowering hints cannot be discarded before LowerTileOp";
    return true;
  }

  static bool IsCodegenKey(const String &key) {
    // Retain keys with actual codegen/outlined-source consumers. A prefix
    // alone does not give a user hint executable semantics, and native host
    // codegen deliberately rejects unlowered TileLang attribute regions.
    return key == "pragma_import_c" || key == "pragma_unroll_factor" ||
           key == "code_block_source" || key == "code_block_signature" ||
           key == "code_block_entry_name" || key == "volatile_scope" ||
           key == "fragment_shape" || key == "fragment_layout" ||
           key == "threadblock_swizzle_pattern" ||
           key == "tl.min_blocks_per_sm";
  }

  static bool RequiresOwningPass(const String &key) {
    // Compiler-owned namespaces are never generic hints. Keep missing-pass
    // diagnostics for reserved internal markers, including future markers,
    // without treating arbitrary user keys as unimplemented contracts.
    if (key.find("tl.") == 0 || key.find("tirx.") == 0)
      return true;
    static const std::unordered_set<std::string> contracts{
        "barrier_init",
        "l2_hit_ratio_map",
        "metal_context",
        "cluster_dims",
        "use_2cta",
        "warp_specialize",
        "kWarpSpecializationScope",
        "kCustomWarpSpecialization",
        "thread_extent",
        "virtual_thread",
        "coproc_scope",
        "pipeline_exec_scope",
        "reduce_scope",
        "vector_count",
        "tilelang_assume",
        "pragma_auto_unroll_max_step",
        "pragma_unroll_explicit",
        "pragma_debug_skip_region"};
    return contracts.count(key);
  }

  SeqStmt Apply(const Map<String, Any> &values, SeqStmt body, Location loc) {
    Map<String, Any> retained;
    std::vector<String> codegen_keys;
    bool explicit_alloc_scope = false;
    for (const auto &[key, value] : values) {
      if (key == "tl.source_block_annotation") {
        TVM_FFI_CHECK(value.as<bool>().value_or(false), ValueError)
            << "tl.source_block_annotation must be the source annotation "
               "marker";
        continue;
      }
      // Native block axes are real scalar Bind/For definitions; reads and
      // writes describe TensorLoad/Store accesses for stage analysis. Their
      // execution semantics are present in the body before this boundary.
      if (key == "tl.axis" || key == "tl.reads" || key == "tl.writes")
        continue;
      if (key == "lexical_alloc_scope") {
        explicit_alloc_scope = true;
        continue;
      }
      if (IsRetired(key))
        continue;
      if (IsCodegenKey(key))
        codegen_keys.push_back(key);
      else if (key == "tl.non_restrict_params")
        retained.Set(key, value);
      else if (RequiresOwningPass(key))
        TVM_FFI_THROW(ValueError)
            << "Unconsumed TileLang annotation `" << key
            << "` at FinalizeAnnotations; run its owning lowering pass first";
      // Other annotations are descriptive user hints. Consuming the metadata
      // leaves its existing sequence and surrounding definition scope intact.
    }
    std::sort(codegen_keys.begin(), codegen_keys.end());
    for (auto it = codegen_keys.rbegin(); it != codegen_keys.rend(); ++it)
      body = SeqStmt(Attribute(Any(), *it, values.at(*it), body, loc));
    if (!retained.empty()) {
      auto encoded = EncodeAnnotations(retained);
      auto call =
          Call(PrimType::Void(), Op::Get("tl.annotation"), {encoded.payload},
               DictAttrs({{"schema_version", 1},
                          {"annotation_schema", encoded.schema}}),
               {}, loc);
      body = SeqStmt({Evaluate(call, loc), body}, loc);
    }
    if (explicit_alloc_scope)
      body = SeqStmt(Lexical(body, loc, true), loc);
    return body;
  }

  bool tile_ops_lowered_;
  bool safety_disabled_;
  bool is_ascend_{false};
  String target_kind_;
  std::unordered_set<std::string> allowed_pending_regions_;
};
} // namespace

namespace transform {
tvm::transform::Pass
FinalizeAnnotations(String target_kind,
                    Array<String> allowed_pending_regions = {}) {
  // This deferral is a scheduling boundary for the two known WS consumers;
  // it cannot suppress unlowered barrier/safety or other internal contracts.
  for (const auto &key : allowed_pending_regions) {
    TVM_FFI_CHECK(key == "kWarpSpecializationScope" ||
                      key == "kCustomWarpSpecialization",
                  ValueError)
        << "FinalizeAnnotations cannot defer region `" << key << "`";
  }
  auto pass = [target_kind,
               allowed_pending_regions](tirx::Function function, IRModule,
                                        tvm::transform::PassContext) {
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return function;
    if (!target_kind.empty()) {
      auto target = function->GetAttr<Target>(tvm::attr::kTarget);
      TVM_FFI_CHECK(target && target.value()->kind->name == target_kind,
                    ValueError)
          << "FinalizeAnnotations target disagrees with Function target";
    }
    auto mutator = ffi::make_object<AnnotationFinalizer>(
        function, allowed_pending_regions);
    auto original = function->body;
    function.CopyOnWrite()->body =
        mutator->Mutate(original).ValueOrUnchanged(original);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.FinalizeAnnotations");
}

tvm::transform::Pass LowerOpaqueBlock() { return FinalizeAnnotations(""); }
} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def_packed(
          "tl.transform.FinalizeAnnotations",
          [](ffi::PackedArgs args, ffi::Any *result) {
            TVM_FFI_CHECK(args.size() == 1 || args.size() == 2, TypeError)
                << "FinalizeAnnotations expects target_kind and optional "
                   "allowed_pending_regions";
            *result = transform::FinalizeAnnotations(
                args[0].cast<String>(), args.size() == 2
                                            ? args[1].cast<Array<String>>()
                                            : Array<String>{});
          })
      .def("tl.transform.LowerOpaqueBlock", transform::LowerOpaqueBlock);
}
} // namespace tvm::tl
