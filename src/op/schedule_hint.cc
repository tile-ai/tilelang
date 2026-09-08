/*!
 * \file tl/op/schedule_hint.cc
 * \brief Marker ops for auto-schedule hints (user-declared facts the dependency
 * analyzer cannot prove; currently `assume_no_conflict`).
 *
 * A no-op marker carrying a user-declared non-conflict hint as a statement:
 *   Evaluate(Call(tl.assume_no_conflict, a, b, level, cross, group))
 * It is consumed by the NormalizeNoConflictHints pass (rewritten into a
 * `no_conflict` For annotation) before AutoSchedule. Registered as `kPure` so
 * that, when auto-schedule is disabled and the marker is left unconsumed,
 * RemoveNoOp drops it -- it never reaches codegen. It has no `TLOpBuilder`:
 * ParseOperator returns an empty TileOperator, which every tile-op consumer
 * skips via `.defined()`.
 */

#include <tvm/ir/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tirx;

TVM_REGISTER_OP("tl.assume_no_conflict")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "assume_no_conflict");

} // namespace tl
} // namespace tvm
