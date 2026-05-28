#ifndef TVM_TL_TRANSFORM_PIPELINE_BODY_ANALYSIS_H_
#define TVM_TL_TRANSFORM_PIPELINE_BODY_ANALYSIS_H_

#include "../common/bind_utils.h"
#include "../common/pipeline_utils.h"
#include "access_analysis.h"

#include "support/check.h"
#include <tvm/tirx/stmt_functor.h>

#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class PipelinePlanningBodyAnalyzer {
public:
  PipelinePlanningBodyAnalyzer(Map<Var, Buffer> buffer_data_to_buffer,
                               Target target)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        target_(std::move(target)) {}

  std::pair<Array<BufferRegion>, Array<BufferRegion>>
  CollectStmtAccessRegions(const Stmt &stmt) const {
    SBlock block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                 /*name_hint=*/"", /*body*/ stmt);
    auto collector = BufferRegionCollector(buffer_data_to_buffer_, target_);
    collector(block);
    return {collector.GetReads(), collector.GetWrites()};
  }

  BufferSet CollectPipelineWriteBuffers(const Array<Stmt> &stmts) const {
    BufferSet write_buffers;
    for (const Stmt &stmt : stmts) {
      auto [_, writes] = CollectStmtAccessRegions(stmt);
      for (const BufferRegion &write : writes) {
        write_buffers.insert(write->buffer);
      }
    }
    return write_buffers;
  }

  bool
  IsReplayableScalarBindStmt(const Stmt &stmt,
                             const BufferSet &pipeline_write_buffers) const {
    auto [reads, _] = CollectStmtAccessRegions(stmt);
    return IsReplayableScalarBind(stmt, reads, pipeline_write_buffers);
  }

  struct ScheduledStmtAnalysis {
    size_t original_stmt_count{0};
    size_t stage_stmt_count{0};
    Array<Stmt> scheduled_stmts;
    std::vector<size_t> scheduled_indices;
    std::vector<size_t> scheduled_stage_indices;
    Array<Integer> replayable_bind_mask;
  };

  ScheduledStmtAnalysis AnalyzeScheduledStmts(const Array<Stmt> &stmts) const {
    BufferSet pipeline_write_buffers = CollectPipelineWriteBuffers(stmts);
    ScheduledStmtAnalysis analysis;
    analysis.original_stmt_count = stmts.size();
    analysis.replayable_bind_mask.reserve(stmts.size());
    size_t stage_stmt_index = 0;
    for (size_t i = 0; i < stmts.size(); ++i) {
      const Stmt &stmt = stmts[i];
      if (IsPipelineDeclarationStmt(stmt)) {
        continue;
      }
      bool replayable =
          IsReplayableScalarBindStmt(stmt, pipeline_write_buffers);
      analysis.replayable_bind_mask.push_back(Integer(replayable ? 1 : 0));
      if (replayable) {
        ++stage_stmt_index;
        continue;
      }
      analysis.scheduled_indices.push_back(i);
      analysis.scheduled_stage_indices.push_back(stage_stmt_index);
      analysis.scheduled_stmts.push_back(stmt);
      ++stage_stmt_index;
    }
    analysis.stage_stmt_count = stage_stmt_index;
    return analysis;
  }

  Array<Integer> FilterAnnotationsForScheduledStmts(
      const Array<Integer> &annotations,
      const ScheduledStmtAnalysis &analysis) const {
    if (annotations.size() == analysis.scheduled_stmts.size()) {
      return annotations;
    }

    Array<Integer> filtered;
    if (annotations.size() == analysis.stage_stmt_count) {
      for (size_t index : analysis.scheduled_stage_indices) {
        filtered.push_back(annotations[index]);
      }
    } else {
      ICHECK_EQ(annotations.size(), analysis.original_stmt_count)
          << "PipelinePlanning: expected pipeline annotation size to match "
             "the scheduled statement count, executable statement count, or "
             "original statement count";
      for (size_t index : analysis.scheduled_indices) {
        filtered.push_back(annotations[index]);
      }
    }
    ICHECK_EQ(filtered.size(), analysis.scheduled_stmts.size());
    return filtered;
  }

  class SeqStmtFlattener : public StmtFunctor<Array<Stmt>(const Stmt &)> {
  public:
    using Base = StmtFunctor<Array<Stmt>(const Stmt &)>;

    static Array<Stmt> Flatten(const Array<Stmt> &stmts) {
      SeqStmtFlattener flattener;
      Array<Stmt> flattened;
      for (const Stmt &stmt : stmts) {
        Array<Stmt> nested = flattener(stmt);
        flattened.insert(flattened.end(), nested.begin(), nested.end());
      }
      return flattened;
    }

    Array<Stmt> VisitStmt(const Stmt &stmt) final {
      if (!stmt.as<SeqStmtNode>()) {
        return Array<Stmt>{stmt};
      }
      return Base::VisitStmt(stmt);
    }

    Array<Stmt> VisitStmt_(const SeqStmtNode *op) final {
      Array<Stmt> flattened;
      for (const Stmt &stmt : op->seq) {
        Array<Stmt> nested = VisitStmt(stmt);
        flattened.insert(flattened.end(), nested.begin(), nested.end());
      }
      return flattened;
    }

    Array<Stmt> VisitStmtDefault_(const Object *) final {
      return Array<Stmt>();
    }
  };

private:
  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_PIPELINE_BODY_ANALYSIS_H_
