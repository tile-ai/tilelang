/*!
 * \file tl/op/schedule_hint.cc
 * \brief Marker op for user-declared auto-schedule conflict facts.
 *
 * One no-op marker carries both conflict polarities as statements:
 *   Evaluate(Call(tl.conflict_hint, a, b, level, cross, group,
 *                 is_conflict))
 * The marker is consumed by NormalizeConflictHints before AutoSchedule.
 * Registered as `kPure` so that, when auto-schedule is disabled and the marker
 * is left unconsumed, RemoveNoOp drops it -- it never reaches codegen. It has
 * no `TLOpBuilder`: ParseOperator returns an empty TileOperator, which every
 * tile-op consumer skips via `.defined()`.
 */

#include <tvm/ir/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tirx;

TVM_REGISTER_OP("tl.conflict_hint")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "conflict_hint");

} // namespace tl
} // namespace tvm
