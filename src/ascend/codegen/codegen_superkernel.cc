#include "codegen_superkernel.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_set>

#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>

#include "ascend/op/builtin.h"
#include "codegen_ascend.h"
#include "support/check.h"

namespace tvm {
namespace codegen {

using namespace tirx;

namespace {

struct SuperKernelConfig {
  bool enabled;
  int num_variants;
  uint64_t cap;
};

SuperKernelConfig ReadConfig() {
  auto ctx = tvm::transform::PassContext::Current();
  bool enabled =
      ctx->GetConfig<Bool>(tl::kEnableSuperKernel, Bool(false)).value();
  bool configured = ctx->config.count(tl::kSuperKernelNumVariants) ||
                    ctx->config.count(tl::kSuperKernelDisableDcci);
  ICHECK(enabled || !configured)
      << "SuperKernel options require tl.ascend_enable_superkernel=True";
  int64_t variants =
      ctx->GetConfig<Integer>(tl::kSuperKernelNumVariants, Integer(4))
          .value()
          ->value;
  ICHECK(variants >= 1 && variants <= 4)
      << "tl.ascend_superkernel_num_variants must be in [1, 4]";
  uint64_t cap =
      ctx->GetConfig<Bool>(tl::kSuperKernelDisableDcci, Bool(false)).value()
          ? uint64_t{4}
          : uint64_t{0};
  return {enabled, static_cast<int>(variants), cap};
}

void ValidateSymbols(const IRModule &mod) {
  std::unordered_set<std::string> symbols;
  for (const auto &kv : mod->functions) {
    auto f = Downcast<PrimFunc>(kv.second);
    if (auto name = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
      symbols.insert(name.value());
    }
  }
  for (const auto &name : symbols) {
    ICHECK(!symbols.count(name + "_sk") && !symbols.count(name + "_sk_args"))
        << "SuperKernel generated symbol collides with an existing global: "
        << name;
  }
}

// Change only the copied SK body. Task-context values become explicit IR vars,
// so normal SIMD/SIMT capture analysis passes them to helpers automatically.
class SkEntryRewriter : public StmtExprMutator {
public:
  SkEntryRewriter(Var index, Var count, bool vector)
      : index_(index), count_(count), vector_(vector) {}

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    // Imported declarations were already emitted with the ordinary entry.
    if (op->attr_key == tirx::attr::pragma_import_c) {
      return VisitStmt(op->body);
    }
    if (vector_ && op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>();
          iv && iv->thread_tag == "blockIdx.x") {
        PrimExpr index = index_;
        if (iv->var.dtype() != index_.dtype()) {
          index = Cast(iv->var.dtype(), index_);
        }
        return Substitute(VisitStmt(op->body), {{iv->var, index}});
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if ((op->op.same_as(builtin::call_extern()) ||
         op->op.same_as(builtin::call_pure_extern())) &&
        op->args.size() == 1) {
      if (const auto *symbol = op->args[0].as<StringImmNode>()) {
        if (symbol->value == "AscendC::GetBlockNum" ||
            symbol->value == "asc_get_block_num") {
          return Cast(op->dtype, count_);
        }
        if (vector_ && (symbol->value == "AscendC::GetBlockIdx" ||
                        symbol->value == "asc_get_block_idx")) {
          return Cast(op->dtype, index_);
        }
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  Var index_, count_;
  bool vector_;
};

} // namespace

// This emitter is the only SK-specific client of ordinary codegen state. It
// reuses type printing, kernel preparation and body emission without adding an
// SK mode to the ordinary generator or its statement/expression visitors.
class SuperKernelCodegen {
public:
  static void Generate(const IRModule &mod, CodeGenTileLangAscend *cg) {
    const auto config = ReadConfig();
    if (!config.enabled) {
      return;
    }
    ValidateSymbols(mod);
    std::ostringstream preamble;
    preamble << "#include <kernel_operator.h>\n" << cg->preamble_stream_.str();
    cg->preamble_stream_.swap(preamble);
    for (const auto &kv : mod->functions) {
      auto f = Downcast<PrimFunc>(kv.second);
      if (f->GetAttr<Integer>(tvm::attr::kCallingConv) ==
          CallingConv::kDeviceKernelLaunch) {
        EmitFunction(f, cg, config);
      }
    }
  }

private:
  static void PrintParameterType(const Var &v, CodeGenTileLangAscend *cg,
                                 std::ostream &os) {
    if (v.dtype().is_handle()) {
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        ICHECK(ptr->storage_scope != "grid_constant")
            << "SuperKernel does not support grid_constant parameters";
      }
      auto it = cg->alloc_storage_scope_.find(v.get());
      if (it != cg->alloc_storage_scope_.end()) {
        cg->PrintStorageScope(it->second, os);
      }
    }
    cg->CodeGenC::PrintType(GetType(v), os);
  }

  static void EmitFunction(const PrimFunc &f, CodeGenTileLangAscend *cg,
                           const SuperKernelConfig &config) {
    cg->PrepareFunction(f);
    const auto name = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).value();
    cg->current_function_name_ = std::string(name) + "_sk";
    for (const char *symbol :
         {"tl_sk_args", "tl_sk_sys", "tl_sk_block_idx", "tl_sk_num_blocks"}) {
      cg->name_supply_->ReserveName(symbol);
    }
    Var index("tl_sk_block_idx", DataType::Int(32));
    Var count("tl_sk_num_blocks", DataType::Int(32));
    cg->var_idmap_[index.get()] = "tl_sk_block_idx";
    cg->var_idmap_[count.get()] = "tl_sk_num_blocks";
    using KernelMode = CodeGenTileLangAscend::AscendKernelMode;
    bool vector = cg->kernel_mode_ == KernelMode::kVector;
    const char *qualifier =
        vector ? "__vector__"
               : (cg->kernel_mode_ == KernelMode::kCube ? "__cube__"
                                                        : "__mix__(1, 2)");
    auto &os = cg->stream;
    os << "struct " << name << "_sk_args {\n";
    for (size_t i = 0; i < f->params.size(); ++i) {
      const Var &v = f->params[i];
      ICHECK(v.dtype().is_scalar())
          << "SuperKernel parameters must be scalar or pointers";
      os << "  ";
      if (!v.dtype().is_handle() && v.dtype().bits() < 32) {
        os << "alignas(4) ";
      }
      PrintParameterType(v, cg, os);
      os << " arg" << i << ";\n";
    }
    os << "};\n\ntemplate<uint32_t tl_sk_variant>\n"
       << "__sk__ " << qualifier << " void " << name << "_sk(const " << name
       << "_sk_args* tl_sk_args, sk::SkSystemArgs* tl_sk_sys) {\n"
       << "  asc_init();\n  tl_sk_sys->SkInit();\n"
       << "  const int32_t tl_sk_num_blocks = tl_sk_sys->SkGetNumBlocks();\n";
    if (vector) {
      os << "  const int32_t tl_sk_block_idx = tl_sk_sys->SkGetBlockIdx();\n";
    }
    bool no_alias = f->HasNonzeroAttr(tirx::attr::kNoAlias);
    for (size_t i = 0; i < f->params.size(); ++i) {
      const Var &v = f->params[i];
      os << "  ";
      PrintParameterType(v, cg, os);
      if (v.dtype().is_handle() && no_alias) {
        cg->PrintRestrict(v, os);
      }
      os << " " << cg->AllocVarID(v.get()) << " = tl_sk_args->arg" << i
         << ";\n";
      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          cg->RegisterHandleType(v.get(), prim->dtype);
        }
      }
    }
    cg->CodeGenTileLangCBase::PreFunctionBody(f);
    int scope = cg->BeginScope();
    cg->PrintStmt(SkEntryRewriter(index, count, vector)(f->body));
    cg->EndScope(scope);
    ICHECK(cg->IsOutsideVF()) << "Ascend VF mode leaked after SK codegen";
    os << "}\n\nSK_BIND(" << name << ", " << config.cap << "ULL";
    for (int i = 0; i < config.num_variants; ++i) {
      os << ", " << name << "_sk<" << i << ">";
    }
    os << ");\n\n";
  }
};

void EmitSuperKernel(const IRModule &mod, CodeGenTileLangAscend *codegen) {
  SuperKernelCodegen::Generate(mod, codegen);
}

} // namespace codegen
} // namespace tvm
