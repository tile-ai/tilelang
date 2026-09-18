/*!
 * \file resolve_sunmmio_unit.cc
 * \brief Resolve the sending ODMA unit for lowered Sunmmio transfer calls.
 */

#include <npuir/SDK/DeviceQuery.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#include "../../op/builtin.h"
#include "../../op/comm.h"
#include "../../op/utils.h"
#include "../sunmmio_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

using npuir::sdk::DeviceQuery;
using NpuMemorySpace = mlir::suvm::MemorySpace;
using NpuUnit = mlir::suvm::Unit;

NpuMemorySpace ToNpuMemorySpace(const Buffer &buffer) {
  const ffi::String &scope = buffer.scope();
  if (scope.empty() || scope == "global") {
    return NpuMemorySpace::global;
  }
  if (scope == kSunmmioScopeRSRAM || scope == "local") {
    return NpuMemorySpace::rsram;
  }
  if (scope == kSunmmioScopeASRAM) {
    return NpuMemorySpace::asram;
  }
  if (scope == kSunmmioScopeWSRAM) {
    return NpuMemorySpace::wsram;
  }
  LOG(FATAL) << "ResolveSunmmioUnit does not recognize memory scope " << scope;
  TVM_FFI_UNREACHABLE();
}

SunmmioOdmaUnit ToTileLangUnit(NpuUnit unit) {
  if (unit == NpuUnit::Odma0) {
    return SunmmioOdmaUnit::kOdma0;
  }
  if (unit == NpuUnit::Odma1) {
    return SunmmioOdmaUnit::kOdma1;
  }
  LOG(FATAL) << "NPU-IR route selected a non-ODMA sending unit";
  TVM_FFI_UNREACHABLE();
}

void AddUniqueUnit(std::vector<NpuUnit> *units, NpuUnit unit) {
  if (std::find(units->begin(), units->end(), unit) == units->end()) {
    units->push_back(unit);
  }
}

class SunmmioUnitResolver : public StmtExprMutator {
public:
  SunmmioUnitResolver() : query_(mlir::suvm::DeviceArch::a4e) {}

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr visited = StmtExprMutator::VisitExpr_(op);
    const auto *call = visited.as<CallNode>();
    ICHECK(call);

    if (!IsOdmaTransfer(call)) {
      return visited;
    }
    std::optional<SunmmioOdmaUnit> existing = GetSunmmioOdmaUnit(call);
    if (existing.has_value()) {
      SunmmioOdmaUnit expected = ResolveUnit(call);
      ICHECK(*existing == expected)
          << "Sunmmio transfer carries " << StringifySunmmioOdmaUnit(*existing)
          << " but A4E routing selects " << StringifySunmmioOdmaUnit(expected);
      return visited;
    }

    SunmmioOdmaUnit unit = ResolveUnit(call);
    Array<PrimExpr> args = call->args;
    args.push_back(MakeSunmmioOdmaUnitExpr(unit));
    return Call(call->dtype, call->op, std::move(args), call->annotations,
                call->span);
  }

  bool IsOdmaTransfer(const CallNode *call) const {
    return call->op.same_as(dma_copy()) ||
           call->op.same_as(sunmmio_layout_transform()) ||
           call->op.same_as(sunmmio_transpose()) ||
           call->op.same_as(broadcast_());
  }

  SunmmioOdmaUnit ResolveUnit(const CallNode *call) const {
    ICHECK_GE(call->args.size(), 2U)
        << "Sunmmio ODMA transfer expects source and destination regions";
    BufferRegion src = NormalizeToBufferRegion(call->args[0]);
    BufferRegion dst = NormalizeToBufferRegion(call->args[1]);
    NpuMemorySpace src_space = ToNpuMemorySpace(src->buffer);
    NpuMemorySpace dst_space = ToNpuMemorySpace(dst->buffer);

    if (call->op.same_as(sunmmio_layout_transform()) ||
        call->op.same_as(sunmmio_transpose())) {
      RequireRoute(src_space, dst_space, NpuUnit::Odma1, std::nullopt);
      return SunmmioOdmaUnit::kOdma1;
    }

    if (call->op.same_as(broadcast_())) {
      ICHECK_GT(call->args.size(), static_cast<size_t>(kBroadcastArgDirection));
      const auto *direction =
          call->args[kBroadcastArgDirection].as<IntImmNode>();
      ICHECK(direction && (direction->value == 0 || direction->value == 1))
          << "tl.broadcast_ requires constant row/col direction before unit "
             "resolution";
      mlir::suvm::McastDirection mcast_direction =
          direction->value == 0 ? mlir::suvm::McastDirection::row
                                : mlir::suvm::McastDirection::col;
      NpuUnit link = query_.mcastLink(mcast_direction);
      return ToTileLangUnit(
          RequireRoute(src_space, dst_space, std::nullopt, link));
    }

    std::vector<NpuUnit> candidates =
        LocalRouteUnits(src_space, dst_space, std::nullopt);
    ICHECK(!candidates.empty())
        << "No A4E ODMA route supports tl.dma_copy from " << src->buffer.scope()
        << " to " << dst->buffer.scope();
    if (candidates.size() == 1) {
      return ToTileLangUnit(candidates.front());
    }

    ICHECK(src_space == NpuMemorySpace::rsram &&
           dst_space == NpuMemorySpace::rsram)
        << "Ambiguous A4E ODMA route outside RSRAM-to-RSRAM copy";
    // ODMA1 handles both contiguous and strided RSRAM-to-RSRAM transfers.
    return SunmmioOdmaUnit::kOdma1;
  }

  std::vector<NpuUnit>
  LocalRouteUnits(NpuMemorySpace src, NpuMemorySpace dst,
                  std::optional<NpuUnit> required_write_unit) const {
    std::vector<NpuUnit> units;
    for (const npuir::sdk::RouteInfo &route : query_.getOdmaRoutes(src, dst)) {
      if (query_.isLinkUnit(route.write.unit)) {
        if (!required_write_unit || route.write.unit != *required_write_unit) {
          continue;
        }
      } else if (required_write_unit) {
        continue;
      }
      AddUniqueUnit(&units, route.odmaUnit);
    }
    return units;
  }

  NpuUnit RequireRoute(NpuMemorySpace src, NpuMemorySpace dst,
                       std::optional<NpuUnit> required_sender,
                       std::optional<NpuUnit> required_write_unit) const {
    std::vector<NpuUnit> units = LocalRouteUnits(src, dst, required_write_unit);
    if (required_sender) {
      units.erase(std::remove_if(
                      units.begin(), units.end(),
                      [&](NpuUnit unit) { return unit != *required_sender; }),
                  units.end());
    }
    ICHECK_EQ(units.size(), 1U)
        << "Expected one A4E ODMA route for the selected transfer";
    return units.front();
  }

  DeviceQuery query_;
};

PrimFunc ResolveSunmmioUnitInFunc(PrimFunc func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  ICHECK(target.defined()) << "ResolveSunmmioUnit expects a bound target";
  auto mcpu = target.value()->GetAttr<ffi::String>("mcpu");
  ICHECK(!mcpu.has_value() || mcpu.value() == "sunmmio-a4e")
      << "ResolveSunmmioUnit only supports the sunmmio-a4e target";

  SunmmioUnitResolver resolver;
  func.CopyOnWrite()->body = resolver(func->body);
  return func;
}

} // namespace

tvm::transform::Pass ResolveSunmmioUnit() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    return ResolveSunmmioUnitInFunc(std::move(func));
  };
  return tir::transform::CreatePrimFuncPass(pass_func, 0,
                                            "tl.ResolveSunmmioUnit", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ResolveSunmmioUnit", ResolveSunmmioUnit);
}

} // namespace tl
} // namespace tvm
