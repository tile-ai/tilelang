/*!
 * \file codegen_mir.cc
 * \brief Optional direct scalar Host IR to MIR compiler.
 */
#include <mir-gen.h>
#include <mir.h>

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/target/codegen.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#include <psapi.h>
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "op/builtin.h"
#include "support/bytes_io.h"
#include "support/check.h"

namespace tvm::tl {
namespace {
using namespace tirx;
using ffi::GetRef;
using ffi::ObjectPtrEqual;
using ffi::ObjectPtrHash;

void *ResolveSymbol(const char *name) {
  // These entry points are already linked; avoid scanning every loaded DLL.
  if (std::string_view(name) == "TVMFFIFunctionCall")
    return reinterpret_cast<void *>(TVMFFIFunctionCall);
  if (std::string_view(name) == "TVMFFIErrorSetRaisedFromCStr")
    return reinterpret_cast<void *>(TVMFFIErrorSetRaisedFromCStr);
#ifdef _WIN32
  std::vector<HMODULE> modules(128);
  DWORD needed;
  for (;;) {
    ICHECK(EnumProcessModules(GetCurrentProcess(), modules.data(),
                              modules.size() * sizeof(HMODULE), &needed));
    if (needed <= modules.size() * sizeof(HMODULE))
      break;
    modules.resize(needed / sizeof(HMODULE));
  }
  for (size_t i = 0; i < needed / sizeof(HMODULE); ++i)
    if (auto address = GetProcAddress(modules[i], name))
      return reinterpret_cast<void *>(address);
  return nullptr;
#else
  return dlsym(RTLD_DEFAULT, name);
#endif
}

MIR_NO_RETURN void MIRError(MIR_error_type_t, const char *format, ...) {
  char message[1024];
  va_list args;
  va_start(args, format);
  vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  LOG(FATAL) << "MIR: " << message;
  std::abort();
}

MIR_type_t Type(DataType dtype) {
  ICHECK_EQ(dtype.lanes(), 1) << "MIR host codegen requires scalar IR";
  if (dtype.is_handle())
    return MIR_T_P;
  if (dtype == DataType::Float(32))
    return MIR_T_F;
  if (dtype == DataType::Float(64))
    return MIR_T_D;
  ICHECK(dtype.is_int() || dtype.is_uint() || dtype.is_bool())
      << "Unsupported MIR dtype " << dtype;
  switch (dtype.bits()) {
  case 1:
    return MIR_T_U8;
  case 8:
    return dtype.is_uint() ? MIR_T_U8 : MIR_T_I8;
  case 16:
    return dtype.is_uint() ? MIR_T_U16 : MIR_T_I16;
  case 32:
    return dtype.is_uint() ? MIR_T_U32 : MIR_T_I32;
  case 64:
    return dtype.is_uint() ? MIR_T_U64 : MIR_T_I64;
  default:
    LOG(FATAL) << "Unsupported MIR dtype " << dtype;
  }
  return MIR_T_UNDEF;
}

struct MIRState {
  MIR_context_t ctx = MIR_init();
  // BSS slots remain relocatable in the cached MIR, unlike embedded pointers.
  std::unordered_map<std::string, MIR_item_t> packed;
  std::unordered_map<std::string, MIR_item_t> functions;
  std::string entry;
  std::string binary;
  std::string native;
  bool generator_initialized = false;
  std::unordered_map<std::string, MIR_item_t> items;
  MIRState() {
    MIR_set_error_func(ctx, MIRError);
    MIR_set_user_data(ctx, this);
  }
  ~MIRState() {
    if (generator_initialized)
      MIR_gen_finish(ctx);
    MIR_finish(ctx);
  }
  void Link(bool load_native = false) {
    for (auto module = DLIST_HEAD(MIR_module_t, *MIR_get_module_list(ctx));
         module; module = DLIST_NEXT(MIR_module_t, module)) {
      for (auto item = DLIST_HEAD(MIR_item_t, module->items); item;
           item = DLIST_NEXT(MIR_item_t, item)) {
        if (item->item_type == MIR_func_item)
          functions[item->u.func->name] = item;
        if (load_native && item->item_type == MIR_import_item)
          MIR_native_builtin(ctx, module, item->u.import_id);
        if (item->item_type == MIR_bss_item &&
            std::string(item->u.bss->name).rfind("__tl_packed_", 0) == 0)
          packed[item->u.bss->name + 12] = item;
      }
      MIR_load_module(ctx, module);
    }
    MIR_link(
        ctx, [](MIR_context_t, MIR_item_t) {},
        [](const char *name) -> void * {
          void *address = ResolveSymbol(name);
          ICHECK(address != nullptr)
              << "MIR cannot resolve external symbol " << name;
          return address;
        });
    for (auto module = DLIST_HEAD(MIR_module_t, *MIR_get_module_list(ctx));
         module; module = DLIST_NEXT(MIR_module_t, module))
      for (auto item = DLIST_HEAD(MIR_item_t, module->items); item;
           item = DLIST_NEXT(MIR_item_t, item))
        if (const char *name = MIR_item_name(ctx, item))
          items[name] = item;
    support::BytesInStream input(native);
    std::string name, code;
    while (input.Read(&name)) {
      uint64_t count;
      ICHECK(input.Read(&code) && input.Read(&count));
      ICHECK_LE(count, native.size() / 16);
      std::vector<MIR_native_reloc_t> locations(count);
      ICHECK(functions.count(name)) << "Unknown native MIR function " << name;
      for (auto &r : locations) {
        std::string symbol;
        uint64_t offset, call_offset;
        ICHECK(input.Read(&offset) && input.Read(&call_offset) &&
               input.Read(&symbol));
        r.offset = offset;
        r.call_offset = call_offset;
        auto it = items.find(symbol);
        r.item = it == items.end()
                     ? MIR_native_builtin(ctx, functions.at(name)->module,
                                          symbol.c_str())
                     : it->second;
        ICHECK(symbol.empty() || r.item != nullptr)
            << "Unknown native MIR symbol " << symbol;
      }
      ICHECK(MIR_load_native(ctx, functions.at(name),
                             reinterpret_cast<const uint8_t *>(code.data()),
                             code.size(), locations.data(),
                             locations.size()) != nullptr);
    }
    // Cache loading never initializes the generator or recompiles a function.
    for (auto &[name, function] : functions) {
      if (load_native) {
        ICHECK(function->u.func->machine_code != nullptr)
            << "Missing native MIR function " << name;
        continue;
      }
      if (!generator_initialized) {
        MIR_gen_init(ctx);
        MIR_gen_set_host_ir(ctx);
        generator_initialized = true;
      }
      ICHECK(MIR_gen_native(
          ctx, function,
          [](MIR_context_t ctx, MIR_item_t function, const uint8_t *code,
             size_t size, const MIR_native_reloc_t *locations, size_t count,
             void *data) {
            auto &state = *static_cast<MIRState *>(data);
            for (size_t i = 0; i < count; ++i) {
              auto item = locations[i].item;
              ICHECK(!item || item->module == function->module)
                  << "Missing native MIR symbol reconstruction";
            }
            support::BytesOutStream output(&state.native);
            output.Write(std::string(MIR_item_name(ctx, function)));
            output.Write(
                std::string(reinterpret_cast<const char *>(code), size));
            output.Write(uint64_t(count));
            for (size_t i = 0; i < count; ++i) {
              output.Write(uint64_t(locations[i].offset));
              output.Write(uint64_t(locations[i].call_offset));
              output.Write(locations[i].item ? std::string(MIR_item_name(
                                                   ctx, locations[i].item))
                                             : std::string());
            }
          },
          this))
          << "Cannot export native MIR function " << name;
    }
    if (generator_initialized)
      MIR_set_gen_interface(ctx, nullptr);
    if (!load_native) {
      MIR_write_module_data_with_func(
          ctx,
          [](MIR_context_t ctx, uint8_t byte) {
            static_cast<MIRState *>(MIR_get_user_data(ctx))
                ->binary.push_back(byte);
            return int(byte);
          },
          DLIST_HEAD(MIR_module_t, *MIR_get_module_list(ctx)));
    }
  }
};

class CodeGenMIR : public ExprFunctor<MIR_op_t(const PrimExpr &)>,
                   public StmtFunctor<void(const Stmt &)> {
public:
  explicit CodeGenMIR(MIRState *state) : state_(state), ctx_(state->ctx) {}

  void Build(const IRModule &mod) {
    MIR_module_t module = MIR_new_module(ctx_, "tilelang_host");
    try {
      for (const auto &[gvar, base] : mod->functions) {
        auto f = Downcast<PrimFunc>(base);
        ICHECK(f->buffer_map.empty()) << "MIR requires lowered, packed Host IR";
        ICHECK(f->params.size() == 4 && f->params[0].dtype().is_handle() &&
               f->params[1].dtype().is_handle() &&
               f->params[2].dtype() == DataType::Int(32) &&
               f->params[3].dtype().is_handle())
            << "MIR requires the TVM-FFI safe-call signature";
        std::string name =
            f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).value();
        std::vector<std::string> names;
        for (size_t i = 0; i < f->params.size(); ++i)
          names.push_back("arg" + std::to_string(i));
        std::vector<MIR_var_t> args;
        for (size_t i = 0; i < f->params.size(); ++i)
          args.push_back({Type(f->params[i].dtype()), names[i].c_str(), 0});
        MIR_type_t result = MIR_T_I32;
        func_ = MIR_new_func_arr(ctx_, name.c_str(), 1, &result, args.size(),
                                 args.data());
        vars_.clear();
        for (size_t i = 0; i < f->params.size(); ++i)
          vars_[f->params[i]] = MIR_new_reg_op(
              ctx_, MIR_reg(ctx_, names[i].c_str(), func_->u.func));
        try {
          Scoped(f->body);
        } catch (...) {
          Emit(MIR_RET, {Int(-1)});
          MIR_finish_func(ctx_);
          throw;
        }
        Emit(MIR_RET, {Int(0)});
        MIR_finish_func(ctx_);
        state_->functions[name] = func_;
        if (f->HasNonzeroAttr(tirx::attr::kIsEntryFunc))
          state_->entry = name;
      }
    } catch (...) {
      MIR_finish_module(ctx_);
      throw;
    }
    MIR_finish_module(ctx_);
    // The binary reader requires call prototypes before their users.
    for (auto &[name, function] : state_->functions) {
      DLIST_REMOVE(MIR_item_t, module->items, function);
      DLIST_APPEND(MIR_item_t, module->items, function);
    }
    state_->Link();
  }

private:
  MIR_op_t Int(int64_t value) { return MIR_new_int_op(ctx_, value); }
  void Emit(MIR_insn_code_t code, std::vector<MIR_op_t> args) {
    MIR_append_insn(ctx_, func_,
                    MIR_new_insn_arr(ctx_, code, args.size(), args.data()));
  }
  MIR_op_t Reg(DataType dtype) {
    MIR_type_t type = Type(dtype);
    if (!dtype.is_float())
      type = MIR_T_I64;
    std::string name = "r" + std::to_string(next_++);
    return MIR_new_reg_op(
        ctx_, MIR_new_func_reg(ctx_, func_->u.func, type, name.c_str()));
  }
  MIR_insn_code_t Move(DataType dtype) {
    return dtype == DataType::Float(32)   ? MIR_FMOV
           : dtype == DataType::Float(64) ? MIR_DMOV
                                          : MIR_MOV;
  }
  MIR_op_t Op(MIR_insn_code_t code, DataType dtype,
              std::vector<MIR_op_t> args) {
    MIR_op_t result = Reg(dtype);
    args.insert(args.begin(), result);
    Emit(code, args);
    return result;
  }
  MIR_op_t Narrow(MIR_op_t value, DataType dtype) {
    if (dtype.is_float() || dtype.is_handle() || dtype.bits() == 64)
      return value;
    if (dtype.bits() == 1)
      return Op(MIR_AND, dtype, {value, Int(1)});
    MIR_insn_code_t code = dtype.bits() == 8    ? MIR_EXT8
                           : dtype.bits() == 16 ? MIR_EXT16
                                                : MIR_EXT32;
    if (dtype.is_uint())
      code = dtype.bits() == 8    ? MIR_UEXT8
             : dtype.bits() == 16 ? MIR_UEXT16
                                  : MIR_UEXT32;
    return Op(code, dtype, {value});
  }
  MIR_op_t Materialize(MIR_op_t value, DataType dtype = DataType::Handle()) {
    return value.mode == MIR_OP_REG ? value : Op(Move(dtype), dtype, {value});
  }
  MIR_op_t Address(MIR_op_t base, MIR_op_t index, size_t stride,
                   size_t offset = 0) {
    if (index.mode == MIR_OP_INT) {
      auto displacement = index.u.i * stride + offset;
      return displacement == 0
                 ? base
                 : Op(MIR_ADD, DataType::Handle(), {base, Int(displacement)});
    }
    auto displacement =
        stride == 1 ? index
                    : Op(MIR_MUL, DataType::Int(64), {index, Int(stride)});
    if (offset != 0)
      displacement =
          Op(MIR_ADD, DataType::Int(64), {displacement, Int(offset)});
    return Op(MIR_ADD, DataType::Handle(), {base, displacement});
  }
  MIR_op_t Memory(MIR_op_t address, DataType dtype) {
    auto base = Materialize(address);
    return MIR_new_mem_op(ctx_, Type(dtype), 0, base.u.reg, 0, 1);
  }
  MIR_op_t Stack(size_t size, size_t alignment = 16) {
    auto raw = Reg(DataType::Handle()), aligned = Reg(DataType::Handle());
    auto result = Reg(DataType::Handle());
    // Static sizes allocate once per invocation, not on every host
    // loop iteration (MIR alloca otherwise lives until the function returns).
    MIR_prepend_insn(ctx_, func_,
                     MIR_new_insn(ctx_, MIR_AND, result, aligned,
                                  Int(-static_cast<int64_t>(alignment))));
    MIR_prepend_insn(
        ctx_, func_,
        MIR_new_insn(ctx_, MIR_ADD, aligned, raw, Int(alignment - 1)));
    MIR_prepend_insn(
        ctx_, func_,
        MIR_new_insn(ctx_, MIR_ALLOCA, raw, Int(size + alignment - 1)));
    return result;
  }
  MIR_op_t Stack(const PrimExpr &size, size_t alignment = 16) {
    if (auto imm = size.as<IntImmNode>()) {
      ICHECK_GE(imm->value, 0);
      return Stack(static_cast<size_t>(imm->value), alignment);
    }
    auto bytes = VisitExpr(size);
    auto &[saved, before] = stack_scopes_.back();
    if (saved.mode != MIR_OP_REG) {
      saved = Reg(DataType::Handle());
      auto start = MIR_new_insn(ctx_, MIR_BSTART, saved);
      // Save at scope entry, even when allocation is in a lazy expression.
      if (before)
        MIR_insert_insn_after(ctx_, func_, before, start);
      else
        MIR_prepend_insn(ctx_, func_, start);
    }
    auto raw =
        Op(MIR_ALLOCA, DataType::Handle(),
           {Op(MIR_ADD, DataType::Int(64), {bytes, Int(alignment - 1)})});
    return Op(MIR_AND, DataType::Handle(),
              {Op(MIR_ADD, DataType::Handle(), {raw, Int(alignment - 1)}),
               Int(-static_cast<int64_t>(alignment))});
  }
  void Scoped(const Stmt &stmt) {
    stack_scopes_.emplace_back(Int(0),
                               DLIST_TAIL(MIR_insn_t, func_->u.func->insns));
    VisitStmt(stmt);
    if (stack_scopes_.back().first.mode == MIR_OP_REG)
      Emit(MIR_BEND, {stack_scopes_.back().first});
    stack_scopes_.pop_back();
  }
  MIR_op_t Label(MIR_label_t label) { return MIR_new_label_op(ctx_, label); }
  void Mark(MIR_label_t label) { MIR_append_insn(ctx_, func_, label); }
  MIR_op_t Choose(const PrimExpr &condition, const PrimExpr &yes,
                  const PrimExpr &no) {
    MIR_op_t result = Reg(yes.dtype());
    auto otherwise = MIR_new_label(ctx_), done = MIR_new_label(ctx_);
    Emit(MIR_BF, {Label(otherwise), VisitExpr(condition)});
    Emit(Move(yes.dtype()), {result, VisitExpr(yes)});
    Emit(MIR_JMP, {Label(done)});
    Mark(otherwise);
    Emit(Move(no.dtype()), {result, VisitExpr(no)});
    Mark(done);
    return result;
  }
  MIR_op_t MinMax(const PrimExpr &a, const PrimExpr &b, bool minimum) {
    auto dtype = a.dtype();
    auto lhs = Materialize(VisitExpr(a), dtype),
         rhs = Materialize(VisitExpr(b), dtype);
    auto code = dtype.is_float() ? (dtype.bits() == 32 ? MIR_FLT : MIR_DLT)
                                 : (dtype.is_uint() ? MIR_ULT : MIR_LT);
    auto result = Reg(dtype),
         done = MIR_new_label_op(ctx_, MIR_new_label(ctx_));
    Emit(Move(dtype), {result, lhs});
    Emit(MIR_BT, {done, Op(code, DataType::Bool(),
                           {minimum ? lhs : rhs, minimum ? rhs : lhs})});
    Emit(Move(dtype), {result, rhs});
    Mark(done.u.label);
    return result;
  }
  MIR_op_t External(const std::string &name, DataType result,
                    const std::vector<DataType> &types,
                    std::vector<MIR_op_t> values) {
    auto ret_type = result.is_void() ? MIR_T_UNDEF : Type(result);
    std::string signature(1, static_cast<char>(ret_type));
    for (auto type : types)
      signature += static_cast<char>(Type(type));
    auto &proto = prototypes_[signature];
    if (!proto) {
      std::vector<MIR_var_t> args;
      std::vector<std::string> names;
      for (size_t i = 0; i < types.size(); ++i)
        names.push_back("p" + std::to_string(i));
      for (size_t i = 0; i < types.size(); ++i)
        args.push_back({Type(types[i]), names[i].c_str(), 0});
      auto proto_name = "proto" + std::to_string(next_++);
      proto =
          MIR_new_proto_arr(ctx_, proto_name.c_str(), result.is_void() ? 0 : 1,
                            &ret_type, args.size(), args.data());
    }
    auto it = externs_.find(name);
    if (it == externs_.end())
      it = externs_.emplace(name, MIR_new_import(ctx_, name.c_str())).first;
    std::vector<MIR_op_t> operands{MIR_new_ref_op(ctx_, proto),
                                   MIR_new_ref_op(ctx_, it->second)};
    auto ret = result.is_void() ? Int(0) : Reg(result);
    if (!result.is_void())
      operands.push_back(ret);
    operands.insert(operands.end(), values.begin(), values.end());
    Emit(MIR_CALL, operands);
    return ret;
  }
  MIR_op_t String(const std::string &value) {
    auto &item = strings_[value];
    if (!item) {
      auto name = "str" + std::to_string(next_++);
      item = MIR_new_data(ctx_, name.c_str(), MIR_T_U8, value.size() + 1,
                          value.c_str());
    }
    return MIR_new_ref_op(ctx_, item);
  }
  MIR_op_t Packed(const std::string &name, MIR_op_t args, MIR_op_t count,
                  MIR_op_t result) {
    auto &slot = state_->packed[name];
    if (!slot)
      slot = MIR_new_bss(ctx_, ("__tl_packed_" + name).c_str(), sizeof(void *));
    auto handle = Op(MIR_MOV, DataType::Handle(),
                     {Memory(MIR_new_ref_op(ctx_, slot), DataType::Handle())});
    return External(packed_call_, DataType::Int(32),
                    {DataType::Handle(), DataType::Handle(), DataType::Int(32),
                     DataType::Handle()},
                    {handle, args, count, result});
  }
  MIR_op_t Field(const Call &call, DataType *dtype) {
    int field = call->args[2].as<IntImmNode>()->value;
    size_t stride = sizeof(DLTensor), offset = 0;
    switch (field) {
#define FIELD(code, member, type)                                              \
  case builtin::code:                                                          \
    offset = offsetof(DLTensor, member);                                       \
    *dtype = type;                                                             \
    break
      FIELD(kDLTensorData, data, DataType::Handle());
      FIELD(kDLTensorShape, shape, DataType::Handle());
      FIELD(kDLTensorStrides, strides, DataType::Handle());
      FIELD(kDLTensorNDim, ndim, DataType::Int(32));
      FIELD(kDLTensorTypeCode, dtype.code, DataType::UInt(8));
      FIELD(kDLTensorTypeBits, dtype.bits, DataType::UInt(8));
      FIELD(kDLTensorTypeLanes, dtype.lanes, DataType::UInt(16));
      FIELD(kDLTensorByteOffset, byte_offset, DataType::UInt(64));
      FIELD(kDLTensorDeviceId, device.device_id, DataType::Int(32));
      FIELD(kDLTensorDeviceType, device.device_type, DataType::Int(32));
#undef FIELD
    case builtin::kDLTensorAddr:
      break;
    case builtin::kInt64ArrayElem:
      stride = 8;
      *dtype = DataType::Int(64);
      break;
    case builtin::kTVMFFIAnyTypeIndex:
      stride = sizeof(TVMFFIAny);
      offset = offsetof(TVMFFIAny, type_index);
      *dtype = DataType::Int(32);
      break;
    case builtin::kTVMFFIAnyZeroPadding:
      stride = sizeof(TVMFFIAny);
      offset = offsetof(TVMFFIAny, zero_padding);
      *dtype = DataType::Int(32);
      break;
    case builtin::kTVMFFIAnyUnionValue:
      stride = sizeof(TVMFFIAny);
      offset = offsetof(TVMFFIAny, v_int64);
      if (!dtype->is_handle())
        *dtype = dtype->is_float() ? DataType::Float(64) : DataType::Int(64);
      break;
    default:
      LOG(FATAL) << "Unsupported MIR struct field " << field;
    }
    auto address = Address(VisitExpr(call->args[0]), VisitExpr(call->args[1]),
                           stride, offset);
    return field == builtin::kDLTensorAddr ? address : Memory(address, *dtype);
  }

  MIR_op_t VisitExpr_(const VarNode *op) final {
    return vars_.at(GetRef<Var>(op));
  }
  MIR_op_t VisitExpr_(const IntImmNode *op) final { return Int(op->value); }
  MIR_op_t VisitExpr_(const FloatImmNode *op) final {
    return op->dtype.bits() == 32 ? MIR_new_float_op(ctx_, op->value)
                                  : MIR_new_double_op(ctx_, op->value);
  }
  MIR_op_t VisitExpr_(const StringImmNode *op) final {
    return String(op->value);
  }
  MIR_op_t VisitExpr_(const LetNode *op) final {
    vars_[op->var] = Materialize(VisitExpr(op->value), op->var.dtype());
    return VisitExpr(op->body);
  }
  MIR_op_t VisitExpr_(const CastNode *op) final {
    if (op->dtype.is_bool())
      return VisitExpr(op->value != make_zero(op->value.dtype()));
    return Convert(VisitExpr(op->value), op->value.dtype(), op->dtype);
  }
  MIR_op_t Convert(MIR_op_t value, DataType from, DataType to) {
    if (from.is_float() && !to.is_float()) {
      if (to == DataType::UInt(64)) {
        auto result = Reg(to);
        auto low = MIR_new_label(ctx_), done = MIR_new_label(ctx_);
        auto boundary = from.bits() == 32 ? MIR_new_float_op(ctx_, 0x1p63f)
                                          : MIR_new_double_op(ctx_, 0x1p63);
        Emit(MIR_BT, {Label(low), Op(from.bits() == 32 ? MIR_FLT : MIR_DLT,
                                     DataType::Bool(), {value, boundary})});
        auto reduced = Op(from.bits() == 32 ? MIR_FSUB : MIR_DSUB, from,
                          {value, boundary});
        auto integer = Op(from.bits() == 32 ? MIR_F2I : MIR_D2I, to, {reduced});
        Emit(MIR_XOR,
             {result, integer, MIR_new_uint_op(ctx_, uint64_t(1) << 63)});
        Emit(MIR_JMP, {Label(done)});
        Mark(low);
        Emit(from.bits() == 32 ? MIR_F2I : MIR_D2I, {result, value});
        Mark(done);
        return result;
      }
      value =
          Op(from.bits() == 32 ? MIR_F2I : MIR_D2I, DataType::Int(64), {value});
    } else if (!from.is_float() && to.is_float()) {
      value = Op(to.bits() == 32 ? (from.is_uint() ? MIR_UI2F : MIR_I2F)
                                 : (from.is_uint() ? MIR_UI2D : MIR_I2D),
                 to, {value});
    } else if (from.is_float() && to.is_float() && from != to) {
      value = Op(to.bits() == 32 ? MIR_D2F : MIR_F2D, to, {value});
    }
    return Narrow(value, to);
  }
#define BINARY(Kind, Signed, Unsigned, Float, Double)                          \
  MIR_op_t VisitExpr_(const Kind##Node *op) final {                            \
    auto dtype = op->a.dtype();                                                \
    auto code = dtype.is_float()  ? (dtype.bits() == 32 ? Float : Double)      \
                : dtype.is_uint() ? Unsigned                                   \
                                  : Signed;                                    \
    ICHECK(code != MIR_INVALID_INSN)                                           \
        << "Unsupported MIR arithmetic " << GetRef<PrimExpr>(op);              \
    auto a = VisitExpr(op->a), b = VisitExpr(op->b);                           \
    return Narrow(Op(code, op->dtype, {a, b}), op->dtype);                     \
  }
  BINARY(Add, MIR_ADD, MIR_ADD, MIR_FADD, MIR_DADD)
  BINARY(Sub, MIR_SUB, MIR_SUB, MIR_FSUB, MIR_DSUB)
  BINARY(Mul, MIR_MUL, MIR_MUL, MIR_FMUL, MIR_DMUL)
  BINARY(Div, MIR_DIV, MIR_UDIV, MIR_FDIV, MIR_DDIV)
  BINARY(EQ, MIR_EQ, MIR_EQ, MIR_FEQ, MIR_DEQ)
  BINARY(NE, MIR_NE, MIR_NE, MIR_FNE, MIR_DNE)
  BINARY(LT, MIR_LT, MIR_ULT, MIR_FLT, MIR_DLT)
  BINARY(LE, MIR_LE, MIR_ULE, MIR_FLE, MIR_DLE)
  BINARY(GT, MIR_GT, MIR_UGT, MIR_FGT, MIR_DGT)
  BINARY(GE, MIR_GE, MIR_UGE, MIR_FGE, MIR_DGE)
#undef BINARY
  MIR_op_t VisitExpr_(const ModNode *op) final {
    auto a = VisitExpr(op->a), b = VisitExpr(op->b);
    if (op->dtype.is_float())
      return External(op->dtype.bits() == 32 ? "fmodf" : "fmod", op->dtype,
                      {op->dtype, op->dtype}, {a, b});
    return Narrow(
        Op(op->dtype.is_uint() ? MIR_UMOD : MIR_MOD, op->dtype, {a, b}),
        op->dtype);
  }
  MIR_op_t VisitExpr_(const MinNode *op) final {
    return MinMax(op->a, op->b, true);
  }
  MIR_op_t VisitExpr_(const MaxNode *op) final {
    return MinMax(op->a, op->b, false);
  }
  MIR_op_t VisitExpr_(const AndNode *op) final {
    return Choose(op->a, op->b, IntImm(DataType::Bool(), 0));
  }
  MIR_op_t VisitExpr_(const OrNode *op) final {
    return Choose(op->a, IntImm(DataType::Bool(), 1), op->b);
  }
  MIR_op_t VisitExpr_(const NotNode *op) final {
    return Op(MIR_EQ, op->dtype, {VisitExpr(op->a), Int(0)});
  }
  MIR_op_t VisitExpr_(const SelectNode *op) final {
    auto condition = VisitExpr(op->condition);
    auto yes = VisitExpr(op->true_value), no = VisitExpr(op->false_value);
    auto result = Reg(op->dtype);
    auto done = MIR_new_label(ctx_);
    Emit(Move(op->dtype), {result, yes});
    Emit(MIR_BT, {Label(done), condition});
    Emit(Move(op->dtype), {result, no});
    Mark(done);
    return result;
  }
  MIR_op_t VisitExpr_(const BufferLoadNode *op) final {
    if (op->predicate)
      return Choose(*op->predicate, BufferLoad(op->buffer, op->indices),
                    make_zero(op->dtype));
    ICHECK_EQ(op->indices.size(), 1U);
    auto address = Address(vars_.at(op->buffer->data),
                           VisitExpr(op->indices[0]), op->dtype.bytes());
    return Op(Move(op->dtype), op->dtype, {Memory(address, op->dtype)});
  }
  MIR_op_t VisitExpr_(const CallNode *op) final {
    const auto &args = op->args;
    if (op->op.same_as(tl::loop_break())) {
      ICHECK(!loops_.empty()) << "MIR loop_break outside a loop";
      auto [done, depth] = loops_.back();
      for (size_t i = stack_scopes_.size(); i > depth; --i)
        if (stack_scopes_[i - 1].first.mode == MIR_OP_REG)
          Emit(MIR_BEND, {stack_scopes_[i - 1].first});
      Emit(MIR_JMP, {Label(done)});
      return Int(0);
    }
    if (op->op.same_as(builtin::ret())) {
      Emit(MIR_RET, {VisitExpr(args[0])});
      return Int(0);
    }
    if (op->op.same_as(builtin::tvm_throw_last_error())) {
      Emit(MIR_RET, {Int(-1)});
      return Int(0);
    }
    if (op->op.same_as(builtin::if_then_else()))
      return Choose(args[0], args[1], args[2]);
    if (op->op.same_as(builtin::isnullptr()))
      return Op(MIR_EQ, DataType::Bool(), {VisitExpr(args[0]), Int(0)});
    if (op->op.same_as(builtin::large_uint_imm()))
      return MIR_new_uint_op(
          ctx_, (uint64_t(Downcast<IntImm>(args[1])->value) << 32) |
                    uint64_t(Downcast<IntImm>(args[0])->value));
    if (op->op.same_as(builtin::bitwise_not()))
      return Narrow(Op(MIR_XOR, op->dtype, {VisitExpr(args[0]), Int(-1)}),
                    op->dtype);
    if (op->op.same_as(builtin::address_of())) {
      auto load = Downcast<BufferLoad>(args[0]);
      ICHECK_EQ(load->indices.size(), 1U);
      return Address(vars_.at(load->buffer->data), VisitExpr(load->indices[0]),
                     load.dtype().bytes());
    }
    if (op->op.same_as(builtin::reinterpret())) {
      ICHECK_EQ(op->dtype.bytes(), args[0].dtype().bytes());
      if (!op->dtype.is_float() && !args[0].dtype().is_float())
        return Narrow(VisitExpr(args[0]), op->dtype);
      auto storage = Stack(op->dtype.bytes());
      Emit(Move(args[0].dtype()),
           {Memory(storage, args[0].dtype()), VisitExpr(args[0])});
      return Op(Move(op->dtype), op->dtype, {Memory(storage, op->dtype)});
    }
    if (op->op.same_as(builtin::handle_add_byte_offset()))
      return Op(MIR_ADD, op->dtype, {VisitExpr(args[0]), VisitExpr(args[1])});
    if (op->op.same_as(builtin::tvm_struct_get())) {
      auto dtype = op->dtype;
      auto field = Field(GetRef<Call>(op), &dtype);
      return Convert(Op(Move(dtype), dtype, {field}), dtype, op->dtype);
    }
    if (op->op.same_as(builtin::tvm_struct_set())) {
      auto dtype = args[3].dtype();
      auto field = Field(GetRef<Call>(op), &dtype);
      auto value = VisitExpr(args[3]);
      Emit(Move(dtype), {field, Convert(value, args[3].dtype(), dtype)});
      return Int(0);
    }
    if (op->op.same_as(builtin::tvm_stack_alloca())) {
      std::string type = args[0].as<StringImmNode>()->value;
      ICHECK(type == "shape" || type == "array" || type == "tvm_ffi_any");
      int stride = type == "shape"   ? 8
                   : type == "array" ? sizeof(DLTensor)
                                     : sizeof(TVMFFIAny);
      return Stack(cast(DataType::Int(64), args[1]) * stride,
                   type == "tvm_ffi_any" ? 128 : 16);
    }
    if (op->op.same_as(tl::tvm_ffi_call_with_result()))
      return Packed(args[0].as<StringImmNode>()->value, VisitExpr(args[1]),
                    VisitExpr(args[2]), VisitExpr(args[3]));
    if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
      auto result = Stack(sizeof(TVMFFIAny));
      Emit(MIR_MOV, {Memory(result, DataType::Int(64)), Int(0)});
      Emit(MIR_MOV,
           {Memory(Address(result, Int(1), 8), DataType::Int(64)), Int(0)});
      int64_t begin = args[2].as<IntImmNode>()->value,
              end = args[3].as<IntImmNode>()->value;
      auto status =
          Packed(args[0].as<StringImmNode>()->value,
                 Address(VisitExpr(args[1]), Int(begin), sizeof(TVMFFIAny)),
                 Int(end - begin), result);
      auto ok = MIR_new_label(ctx_);
      Emit(MIR_BF, {Label(ok), status});
      Emit(MIR_RET, {Int(-1)});
      Mark(ok);
      auto dtype = op->dtype.is_float()    ? DataType::Float(64)
                   : op->dtype.is_handle() ? DataType::Handle()
                                           : DataType::Int(64);
      return Convert(
          Op(Move(dtype), dtype,
             {Memory(Address(result, Int(1), offsetof(TVMFFIAny, v_int64)),
                     dtype)}),
          dtype, op->dtype);
    }
    if (op->op.same_as(builtin::call_extern()) ||
        op->op.same_as(builtin::call_pure_extern())) {
      std::vector<DataType> types;
      std::vector<MIR_op_t> values;
      for (size_t i = 1; i < args.size(); ++i) {
        types.push_back(args[i].dtype());
        values.push_back(VisitExpr(args[i]));
      }
      return External(args[0].as<StringImmNode>()->value, op->dtype, types,
                      values);
    }
    if (const auto *callee = op->op.as<OpNode>()) {
      static auto symbols = Op::GetAttrMap<TGlobalSymbol>("TGlobalSymbol");
      auto callee_op = GetRef<tvm::Op>(callee);
      if (symbols.count(callee_op)) {
        ffi::Array<PrimExpr> external_args{StringImm(symbols[callee_op])};
        for (const auto &arg : args)
          external_args.push_back(arg);
        return VisitExpr(
            Call(op->dtype, builtin::call_extern(), external_args));
      }
    }
    MIR_insn_code_t code = MIR_INVALID_INSN;
    if (op->op.same_as(builtin::bitwise_and()))
      code = MIR_AND;
    if (op->op.same_as(builtin::bitwise_or()))
      code = MIR_OR;
    if (op->op.same_as(builtin::bitwise_xor()))
      code = MIR_XOR;
    if (op->op.same_as(builtin::shift_left()))
      code = MIR_LSH;
    if (op->op.same_as(builtin::shift_right()))
      code = op->dtype.is_uint() ? MIR_URSH : MIR_RSH;
    if (code != MIR_INVALID_INSN)
      return Narrow(
          Op(code, op->dtype, {VisitExpr(args[0]), VisitExpr(args[1])}),
          op->dtype);
    LOG(FATAL) << "Unsupported MIR host intrinsic " << op->op;
    return Int(0);
  }
  void VisitStmt_(const SeqStmtNode *op) final {
    for (const auto &stmt : op->seq)
      VisitStmt(stmt);
  }
  void VisitStmt_(const EvaluateNode *op) final { VisitExpr(op->value); }
  void VisitStmt_(const BindNode *op) final {
    vars_[op->var] = Materialize(VisitExpr(op->value), op->var.dtype());
  }
  void VisitStmt_(const DeclBufferNode *) final {}
  void VisitStmt_(const AllocBufferNode *op) final {
    PrimExpr size = IntImm(DataType::Int(64), op->buffer->dtype.bytes());
    for (const auto &extent : op->buffer->shape)
      size = size * cast(DataType::Int(64), extent);
    vars_[op->buffer->data] =
        Stack(size, std::max(16, op->buffer->data_alignment));
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == "tl.packed_call") {
      std::string previous = packed_call_;
      packed_call_ = op->value.as<StringImmNode>()->value;
      VisitStmt(op->body);
      packed_call_ = previous;
      return;
    }
    ICHECK(op->attr_key == "compute_scope" ||
           op->attr_key == "storage_alignment" || op->attr_key == "device_id" ||
           op->attr_key == "device_type" || op->attr_key == "tl.assume")
        << "Unsupported MIR host attribute " << op->attr_key;
    VisitStmt(op->body);
  }
  void VisitStmt_(const IfThenElseNode *op) final {
    auto otherwise = MIR_new_label(ctx_), done = MIR_new_label(ctx_);
    Emit(MIR_BF, {Label(otherwise), VisitExpr(op->condition)});
    Scoped(op->then_case);
    Emit(MIR_JMP, {Label(done)});
    Mark(otherwise);
    if (op->else_case)
      Scoped(op->else_case.value());
    Mark(done);
  }
  void VisitStmt_(const AssertStmtNode *op) final {
    auto ok = MIR_new_label(ctx_);
    Emit(MIR_BT, {Label(ok), VisitExpr(op->condition)});
    std::string message;
    for (const auto &part : op->message_parts)
      message += part->value;
    External("TVMFFIErrorSetRaisedFromCStr", DataType::Void(),
             {DataType::Handle(), DataType::Handle()},
             {String(op->error_kind->value), String(message)});
    Emit(MIR_RET, {Int(-1)});
    Mark(ok);
  }
  void VisitStmt_(const ForNode *op) final {
    ICHECK(op->kind == ForKind::kSerial || op->kind == ForKind::kUnrolled)
        << "MIR host supports serial loops only";
    auto index = Reg(op->loop_var.dtype());
    vars_[op->loop_var] = index;
    auto start = VisitExpr(op->min), extent = VisitExpr(op->extent);
    auto end = Op(MIR_ADD, op->loop_var.dtype(), {start, extent});
    auto loop = MIR_new_label(ctx_), done = MIR_new_label(ctx_);
    Emit(MIR_MOV, {index, start});
    Mark(loop);
    Emit(op->loop_var.dtype().is_uint() ? MIR_UBGE : MIR_BGE,
         {Label(done), index, end});
    loops_.emplace_back(done, stack_scopes_.size());
    Scoped(op->body);
    loops_.pop_back();
    Emit(MIR_ADD, {index, index, Int(1)});
    Emit(MIR_JMP, {Label(loop)});
    Mark(done);
  }
  void VisitStmt_(const WhileNode *op) final {
    auto loop = MIR_new_label(ctx_), done = MIR_new_label(ctx_);
    Mark(loop);
    Emit(MIR_BF, {Label(done), VisitExpr(op->condition)});
    loops_.emplace_back(done, stack_scopes_.size());
    Scoped(op->body);
    loops_.pop_back();
    Emit(MIR_JMP, {Label(loop)});
    Mark(done);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->predicate) {
      VisitStmt(IfThenElse(*op->predicate,
                           BufferStore(op->buffer, op->value, op->indices)));
      return;
    }
    ICHECK_EQ(op->indices.size(), 1U);
    auto address =
        Address(vars_.at(op->buffer->data), VisitExpr(op->indices[0]),
                op->value.dtype().bytes());
    Emit(Move(op->value.dtype()),
         {Memory(address, op->value.dtype()), VisitExpr(op->value)});
  }

  std::string packed_call_ = "TVMFFIFunctionCall";
  MIRState *state_;
  MIR_context_t ctx_;
  MIR_item_t func_ = nullptr;
  size_t next_ = 0;
  std::unordered_map<Var, MIR_op_t, ObjectPtrHash, ObjectPtrEqual> vars_;
  std::unordered_map<std::string, MIR_item_t> externs_;
  std::unordered_map<std::string, MIR_item_t> prototypes_;
  std::unordered_map<std::string, MIR_item_t> strings_;
  std::vector<std::pair<MIR_op_t, MIR_insn_t>> stack_scopes_;
  std::vector<std::pair<MIR_label_t, size_t>> loops_;
};

class MIRModuleNode final : public ffi::ModuleObj {
public:
  MIRState state;
  const char *kind() const final { return "tilelang_mir"; }
  int GetPropertyMask() const final {
    return ffi::Module::kRunnable | ffi::Module::kBinarySerializable;
  }
  ffi::Bytes SaveToBytes() const final {
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(state.entry);
    stream.Write(state.binary);
    stream.Write(state.native);
    return ffi::Bytes(buffer);
  }
  ffi::String InspectSource(const ffi::String &) const final {
    auto close = [](FILE *file) { fclose(file); };
    std::unique_ptr<FILE, decltype(close)> file(tmpfile(), close);
    ICHECK(file) << "Cannot create MIR source inspection stream";
    MIR_output(state.ctx, file.get());
    auto size = ftell(file.get());
    ICHECK_GE(size, 0);
    rewind(file.get());
    std::string source(size, '\0');
    ICHECK_EQ(fread(source.data(), 1, size, file.get()),
              static_cast<size_t>(size));
    return source;
  }
  ffi::Optional<ffi::Function> GetFunction(const ffi::String &name) final {
    std::string symbol =
        name == "main" ? state.entry
                       : std::string(ffi::symbol::tvm_ffi_symbol_prefix + name);
    auto it = state.functions.find(symbol);
    if (it == state.functions.end())
      return std::nullopt;
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &[packed_name, slot] : state.packed) {
      auto &handle = *static_cast<void **>(slot->addr);
      if (handle == nullptr) {
        TVM_FFI_CHECK_SAFE_CALL(
            TVMBackendGetFuncFromEnv(this, packed_name.c_str(), &handle));
      }
    }
    auto address = reinterpret_cast<TVMFFISafeCallType>(it->second->addr);
    return ffi::Function::FromExternC(
        new ffi::Module(ffi::GetRef<ffi::Module>(this)), address,
        [](void *self) { delete static_cast<ffi::Module *>(self); });
  }

private:
  std::mutex mutex_;
};

} // namespace

TVM_REGISTER_TARGET_KIND("mir", kDLCPU).set_default_keys({"cpu"});
TVM_FFI_STATIC_INIT_BLOCK() {
  auto build = [](IRModule mod) {
    auto node = ffi::make_object<MIRModuleNode>();
    CodeGenMIR(&node->state).Build(mod);
    return ffi::Module(node);
  };
  ffi::reflection::GlobalDef()
      .def("target.build.tilelang_mir",
           [build](IRModule mod, Target) { return build(mod); })
      .def("ffi.Module.load_from_bytes.tilelang_mir",
           [](ffi::Bytes bytes) {
             auto node = ffi::make_object<MIRModuleNode>();
             support::BytesInStream stream(bytes);
             ICHECK(stream.Read(&node->state.entry));
             ICHECK(stream.Read(&node->state.binary));
             ICHECK(stream.Read(&node->state.native));
             std::string_view input(node->state.binary);
             MIR_set_user_data(node->state.ctx, &input);
             MIR_read_with_func(node->state.ctx, [](MIR_context_t ctx) {
               auto &input =
                   *static_cast<std::string_view *>(MIR_get_user_data(ctx));
               if (input.empty())
                 return EOF;
               int byte = static_cast<uint8_t>(input.front());
               input.remove_prefix(1);
               return byte;
             });
             MIR_set_user_data(node->state.ctx, &node->state);
             node->state.Link(true);
             return ffi::Module(node);
           })
      .def("tilelang.ModuleSerialize",
           [](ffi::Module mod) {
             return ffi::Bytes(codegen::SerializeModuleToBytes(mod, false));
           })
      .def("ffi.Module.load_from_file.tlm", [](ffi::String path, ffi::String) {
        std::ifstream file(std::string(path), std::ios::binary);
        ICHECK(file) << "Cannot open module " << path;
        return codegen::DeserializeModuleFromBytes(
            std::string(std::istreambuf_iterator<char>(file), {}));
      });
}
} // namespace tvm::tl
