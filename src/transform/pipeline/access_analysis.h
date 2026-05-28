#ifndef TVM_TL_TRANSFORM_PIPELINE_ACCESS_ANALYSIS_H_
#define TVM_TL_TRANSFORM_PIPELINE_ACCESS_ANALYSIS_H_

#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include "../common/bind_utils.h"
#include "op/operator.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

bool MayConflict(const Region &region1, const Region &region2);

class BufferRegionCollector : public StmtExprVisitor {
public:
  BufferRegionCollector(Map<Var, Buffer> buffer_data_to_buffer, Target target);

  Array<BufferRegion> GetReads() const;
  Array<BufferRegion> GetWrites() const;
  bool GetGlobalCopyPattern() const;
  bool GetTmaCopyPattern() const;
  bool HasNonCopyTileOp() const;

private:
  static bool IsGlobalLikeBuffer(const Buffer &buffer);

  void HandleTileOp(const TileOperator &tile_op);
  void VisitStmt_(const BufferStoreNode *op) final;
  void VisitExpr_(const BufferLoadNode *op) final;
  void VisitExpr_(const CallNode *op) final;
  void VisitStmt_(const IfThenElseNode *op) final;

  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
  Array<BufferRegion> reads_;
  Array<BufferRegion> writes_;
  bool is_global_read_ = false;
  bool under_buffer_store_ = false;
  bool is_global_copy_pattern_ = false;
  bool is_tma_copy_ = false;
  bool has_non_copy_tile_op_ = false;
  bool within_condition_expr_ = false;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_PIPELINE_ACCESS_ANALYSIS_H_
