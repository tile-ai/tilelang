#include "compat/metadata.h"
#include "transform/tirx_passes.h"

#include "backend/common/codegen/attr.h"
#include "native_codegen.h"

#include "compat/expr.h"
#include "config.h"
#include "transform/common/attr.h"

#include <tvm/backend/cuda/attr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <set>
#include <sstream>
#include <unordered_map>

namespace tvm::tl {
namespace {
class ImportCollector : public tvm::tl::StmtExprMutator {
public:
  explicit ImportCollector(bool cuda) : cuda_(cuda) {}
  std::ostringstream source;
  std::set<std::string> headers;
  ffi::Map<ffi::String, int64_t> kernel_attrs;

private:
  ffi::UnchangedOr<PrimExpr> Mutate_(const CastNode *node,
                                     ffi::InplaceMode mode) final {
    if (cuda_) {
      auto lower =
          ffi::Function::GetGlobalRequired("tl.codegen.cuda.lower_cast");
      auto lowered =
          lower(GetRef<PrimExpr>(node)).cast<ffi::Optional<PrimExpr>>();
      if (lowered) {
        return Mutate(*lowered, mode).ValueOrUnchanged(*lowered);
      }
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                 ffi::InplaceMode mode) final {
    if (cuda_ &&
        (node->op.same_as(tirx::call_extern_op()) ||
         node->op.same_as(tirx::call_pure_extern_op())) &&
        !node->args.empty()) {
      if (const auto *symbol = node->args[0].as<StringImmNode>()) {
        std::string name = symbol->value;
        if (name == "debug_print_var" || name == "debug_print_buffer_value" ||
            name == "debug_print_msg") {
          headers.insert("debug.h");
        }
        if (name.starts_with("tl::") || name == "DP4A") {
          headers.insert("common.h");
          if (name.find("Reduce") != std::string::npos ||
              name.find("reduce") != std::string::npos)
            headers.insert("reduce.h");
          if (name.find("Scan") != std::string::npos ||
              name.find("scan") != std::string::npos ||
              name.starts_with("tl::Cum"))
            headers.insert("scan.h");
          if (name.find("atomic") != std::string::npos)
            headers.insert("atomic.h");
          if (name.find("copy") != std::string::npos)
            headers.insert("copy.h");
          if (name.find("ldmatrix") != std::string::npos ||
              name.find("stmatrix") != std::string::npos)
            headers.insert("ldsm.h");
          if (name.starts_with("tl::mma_sync<"))
            headers.insert("instruction/mma.h");
          if (name.starts_with("tl::sm120_mma_sync_blockscaled<"))
            headers.insert("instruction/mma_block_scale.h");
          if (name.starts_with("tl::wgmma_"))
            headers.insert("instruction/wgmma.h");
          if (name.starts_with("tl::tcgen05mma_"))
            headers.insert("instruction/tcgen05mma.h");
          if (name.starts_with("tl::random_"))
            headers.insert("random.h");
          if (name.find("swizzle") != std::string::npos)
            headers.insert("threadblock_swizzle.h");
        }
      }
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    For loop = GetRef<For>(node);
    if (auto factor = node->annotations.Get("pragma_unroll_factor")) {
      loop.CopyOnWrite()->annotations.Set(tirx::attr::kPragmaUnroll, *factor);
      loop.CopyOnWrite()->annotations.erase("pragma_unroll_factor");
    }
    if (auto threads = node->annotations.Get("tl.cpu_num_threads")) {
      loop.CopyOnWrite()->annotations.Set("codegen.num_threads", *threads);
      loop.CopyOnWrite()->annotations.erase("tl.cpu_num_threads");
    }
    return tvm::tl::StmtExprMutator::Mutate_(loop.get(), mode)
        .ValueOrUnchanged(loop);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    RegionStmt region = GetRef<RegionStmt>(node);
    auto attributes = node->attrs->dict;
    if (cuda_) {
      if (auto minimum = attributes.Get("tl.min_blocks_per_sm")) {
        auto value = GetIntAnnotation(*minimum);
        TVM_FFI_CHECK(value && *value > 0, ValueError)
            << "min_blocks_per_sm must be a positive integer";
        auto previous = kernel_attrs.Get("min_blocks_per_sm");
        TVM_FFI_CHECK(!previous || *previous == *value, ValueError)
            << "Conflicting min_blocks_per_sm annotations in one kernel";
        kernel_attrs.Set("min_blocks_per_sm", *value);
        attributes.erase("tl.min_blocks_per_sm");
        region.CopyOnWrite()->attrs = DictAttrs(attributes);
      }
    }
    if (auto imported = attributes.Get("pragma_import_c")) {
      source << imported->cast<ffi::String>() << "\n\n";
      attributes.erase("pragma_import_c");
      region.CopyOnWrite()->attrs = DictAttrs(attributes);
    }
    return tvm::tl::StmtExprMutator::Mutate_(region.get(), mode)
        .ValueOrUnchanged(region);
  }
  bool cuda_;
};
} // namespace

IRModule PrepareNativeCodegen(IRModule module, const Target &target) {
  std::unordered_map<std::string, std::string> symbols;
  std::ostringstream imports;
  std::set<std::string> headers;
  for (const auto &[global, base] : module->functions) {
    auto function = base.as_or_throw<tirx::Function>();
    auto symbol =
        function->GetAttr(tvm::attr::kGlobalSymbol).value_or(global->name_hint);
    auto [previous, inserted] = symbols.emplace(symbol, global->name_hint);
    TVM_FFI_CHECK(inserted, ValueError)
        << "Duplicate device global_symbol `" << symbol << "` on functions `"
        << previous->second << "` and `" << global->name_hint << "`";
    function = WithAttr(std::move(function), tvm::attr::kGlobalSymbol, symbol);
    for (const auto &[old_key, new_key] :
         {std::pair{"tl.readonly_param_indices",
                    codegen_attr::kReadOnlyParamIndices.name},
          std::pair{"tl.non_restrict_param_indices",
                    codegen_attr::kNonRestrictParamIndices.name},
          std::pair{attr::kCodeBlockSource, codegen_attr::kSource.name}}) {
      if (auto value = function->attrs->dict.Get(old_key)) {
        function = WithAttr(std::move(function), new_key, *value);
      }
    }
    if (function->HasNonzeroAttr(attr::kHasGridSync)) {
      function = WithAttr(std::move(function), tirx::attr::kNoAlias, false);
    }
    if (auto entry =
            function->GetAttr<ffi::String>(attr::kCodeBlockEntryName)) {
      TVM_FFI_CHECK(*entry == symbol, ValueError)
          << "Source-code kernel entry_name must match its lowered "
             "global_symbol";
    }
    if (auto imported = function->GetAttr<ffi::String>("pragma_import_c")) {
      imports << *imported << "\n\n";
      function = WithoutAttr(std::move(function), "pragma_import_c");
    }
    if (function->body) {
      auto collector =
          ffi::make_object<ImportCollector>(target->kind->name == "cuda");
      SeqStmt body = *function->body;
      function.CopyOnWrite()->body =
          collector->Mutate(body).ValueOrUnchanged(body);
      imports << collector->source.str();
      headers.insert(collector->headers.begin(), collector->headers.end());
      if (!collector->kernel_attrs.empty()) {
        auto kernel_attrs =
            function->GetAttr(tvm::backend::cuda::attr::kKernelAttrs)
                .value_or(ffi::Map<ffi::String, int64_t>{});
        for (const auto &[key, value] : collector->kernel_attrs) {
          auto previous = kernel_attrs.Get(key);
          TVM_FFI_CHECK(!previous || *previous == value, ValueError)
              << "Conflicting CUDA kernel attribute: " << key;
          kernel_attrs.Set(key, value);
        }
        function =
            WithAttr(std::move(function),
                     tvm::backend::cuda::attr::kKernelAttrs, kernel_attrs);
      }
    }
    module.CopyOnWrite()->functions.Set(global, function);
  }
  module = WithAttr(std::move(module), codegen_attr::kEmitLineDirectives,
                    tl_config::EmitLineDirectivesEnabled());
  ffi::String preamble = module->GetAttr(codegen_attr::kPreamble).value_or("");
  std::ostringstream includes;
  if (target->kind->name == "cuda" && !imports.str().empty()) {
    includes << "#include <tl_templates/cuda/common.h>\n";
  }
  for (const auto &header : headers)
    includes << "#include <tl_templates/cuda/" << header << ">\n";
  return WithAttr(
      std::move(module), codegen_attr::kPreamble,
      ffi::String(includes.str() + std::string(preamble) + imports.str()));
}

ffi::Module BuildNativeCodegen(IRModule module, Target target,
                               const ffi::String &postproc, bool source_only) {
  module = PrepareNativeCodegen(std::move(module), target);
  if (!postproc.empty() && ffi::Function::GetGlobal(postproc)) {
    module = WithAttr(std::move(module), codegen_attr::kPostproc, postproc);
  }
  module = WithAttr(std::move(module), codegen_attr::kSourceOnly, source_only);
  module = tirx_passes::StmtSimplify()(module);
  if (tvm::transform::PassContext::Current()
          ->GetConfig<bool>("tirx.disable_assert")
          .value_or(false))
    module = tirx::transform::SkipAssert()(module);
  if (auto build =
          ffi::Function::GetGlobal("tl.codegen.build_" + target->kind->name))
    return (*build)(module, target).cast<ffi::Module>();
  return tvm::codegen::Build(std::move(module), std::move(target));
}
} // namespace tvm::tl

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def(
      "target.build.tilelang_webgpu",
      [](tvm::IRModule module, tvm::Target target) {
        return tvm::tl::BuildNativeCodegen(std::move(module),
                                           std::move(target));
      });
}
