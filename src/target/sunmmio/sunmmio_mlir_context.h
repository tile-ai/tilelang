#ifndef TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_
#define TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_

#include "sunmmio_mlir_type.h"

#include "../../layout/layout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "npuir/Dialect/SUVM/IR/Attributes.h"
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace codegen {

struct SunmmioMlirContext {
  SunmmioMlirContext();

  using TirLayoutMap = ffi::Map<tir::Buffer, tl::Layout>;
  // Empty and combined states are masks, not named SyncUnits enumerators.
  using SyncUnitMask = uint32_t;
  static constexpr SyncUnitMask kNoSyncUnits = 0U;

  mlir::MLIRContext mlir_ctx;
  mlir::OpBuilder builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;

  using MLIRValueTable = std::unordered_map<std::string, mlir::Value>;
  std::vector<MLIRValueTable> mlir_value_table_stack;

  std::unordered_map<std::string, mlir::Value> barrier_by_mask;
  std::unordered_map<int64_t, mlir::Value> static_barrier_by_mask;
  SyncUnitMask pending_sync_units{kNoSyncUnits};

  static SyncUnitMask ToSyncUnitMask(mlir::suvm::SyncUnits units) {
    return static_cast<SyncUnitMask>(units);
  }

  static mlir::suvm::SyncUnits ToSyncUnits(SyncUnitMask mask) {
    std::optional<mlir::suvm::SyncUnits> units =
        mlir::suvm::symbolizeSyncUnits(mask);
    ICHECK(units) << "Invalid SUVM synchronization unit mask: " << mask;
    return *units;
  }

  static SyncUnitMask MergeSyncUnits(SyncUnitMask lhs, SyncUnitMask rhs) {
    return lhs | rhs;
  }

  void AddPendingSyncUnits(mlir::suvm::SyncUnits units) {
    pending_sync_units =
        MergeSyncUnits(pending_sync_units, ToSyncUnitMask(units));
  }

  void CompletePendingSyncUnits(SyncUnitMask units) {
    pending_sync_units &= ~units;
  }

  struct ForFrame {
    mlir::scf::ForOp op;
    ffi::Map<ffi::String, ffi::Any> annotations;
    std::vector<std::string> live_out_value_names;
    std::vector<mlir::Value> iter_values;
    std::vector<mlir::Value> produced_values;
    SyncUnitMask entry_pending_sync_units{kNoSyncUnits};
  };
  std::vector<ForFrame> for_stack;
  std::vector<TirLayoutMap> layout_map_stack;
  std::vector<TirLayoutMap> global_layout_map_stack;

  struct WhileFrame {
    mlir::scf::WhileOp op;
    bool in_body{false};
    std::vector<std::string> live_out_value_names;
    std::vector<mlir::Value> before_values;
    std::vector<mlir::Value> iter_values;
    std::vector<mlir::Value> produced_values;
    SyncUnitMask entry_pending_sync_units{kNoSyncUnits};
    SyncUnitMask condition_pending_sync_units{kNoSyncUnits};
  };
  std::vector<WhileFrame> while_stack;

  struct IfFrame {
    mlir::scf::IfOp op;
    bool in_else{false};
    std::vector<std::string> live_out_value_names;
    std::vector<mlir::Value> base_values;
    std::vector<mlir::Value> produced_values;
    std::vector<mlir::Value> then_yield_values;
    SyncUnitMask entry_pending_sync_units{kNoSyncUnits};
    SyncUnitMask then_pending_sync_units{kNoSyncUnits};
  };
  std::vector<IfFrame> if_stack;

  enum class ControlKind { kFor, kIf, kWhile };

  struct ControlNode {
    ControlKind kind;
    int index{0};
  };
  std::vector<ControlNode> control_flow_stack;

  const ffi::Map<ffi::String, ffi::Any> *CurrentForAnnotations() const {
    if (for_stack.empty()) {
      return nullptr;
    }
    return &for_stack.back().annotations;
  }

  void ClearFunctionState() {
    mlir_value_table_stack.clear();
    barrier_by_mask.clear();
    static_barrier_by_mask.clear();
    pending_sync_units = kNoSyncUnits;
    for_stack.clear();
    if_stack.clear();
    while_stack.clear();
    control_flow_stack.clear();
  }

  void ClearMLIRValueScopes() { mlir_value_table_stack.clear(); }

  void PushMLIRValueScope() { mlir_value_table_stack.emplace_back(); }

  void PopMLIRValueScope() {
    if (!mlir_value_table_stack.empty()) {
      mlir_value_table_stack.pop_back();
    }
  }

  mlir::Value LookupMLIRValue(const std::string &name) const {
    for (auto it = mlir_value_table_stack.rbegin();
         it != mlir_value_table_stack.rend(); ++it) {
      auto vit = it->find(name);
      if (vit != it->end()) {
        return vit->second;
      }
    }
    return mlir::Value();
  }

  mlir::Value LookupValue(const SunMMIOValue &value,
                          const std::string &debug_tag) const {
    mlir::Value existing = LookupMLIRValue(value.value);
    if (existing) {
      return existing;
    }

    LOG(FATAL) << "Missing MLIR value for SunMMIO operand `"
               << (value.value.empty() ? std::string("<unnamed>") : value.value)
               << "` while lowering " << debug_tag;
    TVM_FFI_UNREACHABLE();
  }

  void BindMLIRValue(const std::string &name, mlir::Value v) {
    ICHECK(!name.empty()) << "Cannot bind unnamed MLIR value";
    ICHECK(v) << "Cannot bind null MLIR value for `" << name << "`";
    if (mlir_value_table_stack.empty()) {
      mlir_value_table_stack.emplace_back();
    }
    mlir_value_table_stack.back()[name] = v;
    for (auto it = control_flow_stack.rbegin(); it != control_flow_stack.rend();
         ++it) {
      if (it->kind == ControlKind::kFor) {
        ForFrame &frame = for_stack[it->index];
        auto vit = std::find(frame.live_out_value_names.begin(),
                             frame.live_out_value_names.end(), name);
        if (vit == frame.live_out_value_names.end()) {
          continue;
        }
        int idx = static_cast<int>(
            std::distance(frame.live_out_value_names.begin(), vit));
        if (idx >= 0 && idx < static_cast<int>(frame.produced_values.size())) {
          frame.produced_values[idx] = v;
        }
        break;
      }
      if (it->kind == ControlKind::kWhile) {
        WhileFrame &frame = while_stack[it->index];
        if (!frame.in_body) {
          continue;
        }
        auto vit = std::find(frame.live_out_value_names.begin(),
                             frame.live_out_value_names.end(), name);
        if (vit == frame.live_out_value_names.end()) {
          continue;
        }
        int idx = static_cast<int>(
            std::distance(frame.live_out_value_names.begin(), vit));
        if (idx >= 0 && idx < static_cast<int>(frame.produced_values.size())) {
          frame.produced_values[idx] = v;
        }
        break;
      }
      IfFrame &frame = if_stack[it->index];
      auto vit = std::find(frame.live_out_value_names.begin(),
                           frame.live_out_value_names.end(), name);
      if (vit == frame.live_out_value_names.end()) {
        continue;
      }
      int idx = static_cast<int>(
          std::distance(frame.live_out_value_names.begin(), vit));
      if (idx >= 0 && idx < static_cast<int>(frame.produced_values.size())) {
        frame.produced_values[idx] = v;
      }
      break;
    }
  }

  void ClearLayoutScopes();
  void PushLayoutScope(const TirLayoutMap &layout_map,
                       const TirLayoutMap &global_layout_map);
  void PopLayoutScope();
  ffi::Optional<tl::Layout> LookupLayout(const tir::Buffer &buffer) const;
  void ApplyLayoutToType(const tir::Buffer &buffer, SunMMIOType *type) const;

  void Clear();
};

} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_SUNMMIO_MLIR_CONTEXT_H_
