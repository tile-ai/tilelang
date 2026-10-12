#include <gtest/gtest.h>

#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt.h>

#include "transform/common/attr.h"
#include "transform/sunmmio_tile_loop_fusion/discovery.h"
#include "transform/sunmmio_tile_loop_fusion/planner.h"

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace {

using namespace tir;

PrimExpr I(int value) { return IntImm(DataType::Int(32), value); }

Buffer MakeSharedBuffer(const std::string &name, const Array<PrimExpr> &shape) {
  return decl_buffer(shape, DataType::Float(32), name, "shared.rsram");
}

Array<PrimExpr> MakeExtents(std::initializer_list<int> values) {
  Array<PrimExpr> extents;
  for (int value : values) {
    extents.push_back(I(value));
  }
  return extents;
}

Map<String, ObjectRef> Make1DScopeEntryAnnotations() {
  Map<String, ObjectRef> annotations;
  annotations.Set(attr::tile_scope_entry, Integer(1));
  annotations.Set(attr::tile_execution_axis, Integer(0));
  annotations.Set(attr::tile_execution_domain_axes, Array<PrimExpr>{I(0)});
  annotations.Set(attr::kTileDomain, Array<PrimExpr>{I(32)});
  annotations.Set(attr::tile_tile_size, Array<PrimExpr>{I(32)});
  return annotations;
}

Stmt Make1DTileCopy(const Buffer &dst, const Buffer &src,
                    const std::string &axis_name) {
  Var axis(axis_name, DataType::Int(32));
  Stmt body = BufferStore(dst, BufferLoad(src, {axis}), {axis});
  return For(axis, 0, I(32), ForKind::kSerial, body, Optional<IterVar>(),
             Make1DScopeEntryAnnotations());
}

PrimFunc MakeLongRowChainPrimFunc(int num_regions) {
  std::vector<Buffer> buffers;
  buffers.reserve(num_regions + 1);
  for (int i = 0; i <= num_regions; ++i) {
    buffers.push_back(MakeSharedBuffer("buffer_" + std::to_string(i), {I(32)}));
  }

  Array<Stmt> seq;
  for (int i = 0; i < num_regions; ++i) {
    seq.push_back(Make1DTileCopy(buffers[i + 1], buffers[i], "i"));
  }
  return PrimFunc(Array<Var>{}, SeqStmt(seq));
}

BufferRegion MakeFullBufferRegion(const Buffer &buffer) {
  Array<Range> region;
  for (const PrimExpr &extent : buffer->shape) {
    region.push_back(Range::FromMinExtent(I(0), extent));
  }
  return BufferRegion(buffer, region);
}

TileScopeRegion MakePlannerRegion(int global_region_index,
                                  std::vector<std::string> axes,
                                  const Array<PrimExpr> &extents) {
  TileScopeRegion region;
  region.global_region_index = global_region_index;
  region.logical_execution_axis_keys = std::move(axes);
  region.execution_loop_extents = extents;
  return region;
}

NormalizedBufferAccess MakeAccess(const Buffer &buffer, int home_depth,
                                  int64_t payload_bytes) {
  NormalizedBufferAccess access;
  access.region = MakeFullBufferRegion(buffer);
  access.home_depth = home_depth;
  access.payload_bytes = payload_bytes;
  return access;
}

TileScopeDependenceEdge MakeRawEdge(int src_region_index, int dst_region_index,
                                    int src_access_index, int dst_access_index,
                                    int rho, int64_t weight_bytes,
                                    int max_shared_execution_depth = -1) {
  TileScopeDependenceEdge edge;
  edge.src_region_index = src_region_index;
  edge.dst_region_index = dst_region_index;
  edge.kind = TileScopeDependenceKind::kRAW;
  edge.src_access_index = src_access_index;
  edge.dst_access_index = dst_access_index;
  edge.rho = rho;
  edge.max_shared_execution_depth =
      max_shared_execution_depth < 0 ? rho : max_shared_execution_depth;
  edge.weight_bytes = weight_bytes;
  return edge;
}

SunmmioTileLoopFusionWindowPlan
PlanSingleProblem(const SunmmioTileLoopFusionWindowProblem &problem) {
  std::vector<SunmmioTileLoopFusionWindowPlan> plans =
      PlanSunmmioTileLoopFusionWindowProblems({problem});
  EXPECT_EQ(plans.size(), 1U);
  return plans[0];
}

const SunmmioTileLoopFusionPlannerTreeNode &
ExpectSingleScopeChild(const SunmmioTileLoopFusionPlannerTreeNode &node) {
  EXPECT_EQ(node.children.size(), 1U);
  EXPECT_TRUE(node.children[0].is_scope);
  return node.children[0];
}

void CollectLeafRegionOrder(
    const std::vector<SunmmioTileLoopFusionPlannerTreeNode> &nodes,
    std::vector<int> *order) {
  for (const SunmmioTileLoopFusionPlannerTreeNode &node : nodes) {
    if (node.is_scope) {
      CollectLeafRegionOrder(node.children, order);
    } else {
      order->push_back(node.region_index);
    }
  }
}

std::vector<int> LeafRegionOrder(
    const std::vector<SunmmioTileLoopFusionPlannerTreeNode> &nodes) {
  std::vector<int> order;
  CollectLeafRegionOrder(nodes, &order);
  return order;
}

TEST(SunmmioTileLoopFusionPlannerTest, LargeWindowFallsBackToSourceOrder) {
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(MakeLongRowChainPrimFunc(19));
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);
  std::vector<SunmmioTileLoopFusionWindowPlan> plans =
      PlanSunmmioTileLoopFusionWindowProblems(problems);

  ASSERT_EQ(program.region_runs.size(), 1U);
  EXPECT_EQ(program.region_runs[0].num_regions, 19);
  ASSERT_EQ(plans.size(), 1U);
  const SunmmioTileLoopFusionWindowPlan &plan = plans[0];

  EXPECT_EQ(LeafRegionOrder(plan.tree),
            std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                              15, 16, 17, 18}));

  EXPECT_GT(plan.score.write_cut_cost, 0);
  ASSERT_EQ(plan.tree.size(), 19U);
  for (int i = 0; i < 19; ++i) {
    EXPECT_FALSE(plan.tree[i].is_scope);
    EXPECT_EQ(plan.tree[i].region_index, i);
  }
}

TEST(SunmmioTileLoopFusionPlannerTest,
     BoundaryWindowOf15UsesExactPlannerAndKeepsRawReuse) {
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(MakeLongRowChainPrimFunc(15));
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);
  std::vector<SunmmioTileLoopFusionWindowPlan> plans =
      PlanSunmmioTileLoopFusionWindowProblems(problems);

  ASSERT_EQ(program.region_runs.size(), 1U);
  EXPECT_EQ(program.region_runs[0].num_regions, 15);
  ASSERT_EQ(plans.size(), 1U);
  const SunmmioTileLoopFusionWindowPlan &plan = plans[0];

  EXPECT_EQ(
      LeafRegionOrder(plan.tree),
      std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}));
  EXPECT_EQ(plan.score.write_cut_cost, 0);
  // The first input read is unavoidable; all intermediate RAW reads are reused.
  EXPECT_EQ(plan.score.shared_read_cost, 128);
  EXPECT_EQ(plan.score.reorder_penalty, 0);

  ASSERT_EQ(plan.tree.size(), 1U);
  EXPECT_TRUE(plan.tree[0].is_scope);
  EXPECT_EQ(plan.tree[0].shell_axes, std::vector<std::string>({"i"}));
  ASSERT_EQ(plan.tree[0].children.size(), 15U);
  for (int i = 0; i < 15; ++i) {
    EXPECT_FALSE(plan.tree[0].children[i].is_scope);
    EXPECT_EQ(plan.tree[0].children[i].region_index, i);
  }
}

TEST(SunmmioTileLoopFusionPlannerTest,
     WindowOf16RegionsFallsBackAtExactPlannerBoundary) {
  SunmmioTileLoopFusionProgram program =
      BuildSunmmioTileLoopFusionProgram(MakeLongRowChainPrimFunc(16));
  std::vector<SunmmioTileLoopFusionWindowProblem> problems =
      BuildSunmmioTileLoopFusionWindowProblems(program);
  std::vector<SunmmioTileLoopFusionWindowPlan> plans =
      PlanSunmmioTileLoopFusionWindowProblems(problems);

  ASSERT_EQ(program.region_runs.size(), 1U);
  EXPECT_EQ(program.region_runs[0].num_regions, 16);
  ASSERT_EQ(plans.size(), 1U);
  const SunmmioTileLoopFusionWindowPlan &plan = plans[0];

  EXPECT_EQ(
      LeafRegionOrder(plan.tree),
      std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}));
  EXPECT_GT(plan.score.write_cut_cost, 0);
  ASSERT_EQ(plan.tree.size(), 16U);
  for (int i = 0; i < 16; ++i) {
    EXPECT_FALSE(plan.tree[i].is_scope);
    EXPECT_EQ(plan.tree[i].region_index, i);
  }
}

TEST(SunmmioTileLoopFusionPlannerTest, SharedRowRawEdgeBuildsSingleOuterShell) {
  Buffer row_buffer = MakeSharedBuffer("row_buffer", {I(8)});

  SunmmioTileLoopFusionWindowProblem problem;
  problem.regions = {
      MakePlannerRegion(0, {"i"}, MakeExtents({8})),
      MakePlannerRegion(1, {"i"}, MakeExtents({8})),
  };
  problem.normalized_regions.resize(2);
  problem.normalized_regions[0].def_out = {MakeAccess(row_buffer, 1, 32)};
  problem.normalized_regions[1].use_in = {MakeAccess(row_buffer, 1, 32)};
  problem.graph.edges = {MakeRawEdge(0, 1, 0, 0, 1, 32)};

  SunmmioTileLoopFusionWindowPlan plan = PlanSingleProblem(problem);

  EXPECT_EQ(plan.region_indices, std::vector<int>({0, 1}));
  EXPECT_EQ(LeafRegionOrder(plan.tree), std::vector<int>({0, 1}));
  EXPECT_EQ(plan.score.write_cut_cost, 0);
  EXPECT_EQ(plan.score.shared_read_cost, 0);
  EXPECT_EQ(plan.score.reorder_penalty, 0);

  ASSERT_EQ(plan.tree.size(), 1U);
  EXPECT_TRUE(plan.tree[0].is_scope);
  EXPECT_EQ(plan.tree[0].shell_axes, std::vector<std::string>({"i"}));
  ASSERT_EQ(plan.tree[0].children.size(), 2U);
  EXPECT_FALSE(plan.tree[0].children[0].is_scope);
  EXPECT_EQ(plan.tree[0].children[0].region_index, 0);
  EXPECT_FALSE(plan.tree[0].children[1].is_scope);
  EXPECT_EQ(plan.tree[0].children[1].region_index, 1);
}

TEST(SunmmioTileLoopFusionPlannerTest, SharedTileRawEdgeBuildsNestedTileShell) {
  Buffer tile_buffer = MakeSharedBuffer("tile_buffer", {I(8), I(1)});

  SunmmioTileLoopFusionWindowProblem problem;
  problem.regions = {
      MakePlannerRegion(0, {"i", "j"}, MakeExtents({8, 1})),
      MakePlannerRegion(1, {"i", "j"}, MakeExtents({8, 1})),
  };
  problem.normalized_regions.resize(2);
  problem.normalized_regions[0].def_out = {MakeAccess(tile_buffer, 2, 128)};
  problem.normalized_regions[1].use_in = {MakeAccess(tile_buffer, 2, 128)};
  problem.graph.edges = {MakeRawEdge(0, 1, 0, 0, 2, 128)};

  SunmmioTileLoopFusionWindowPlan plan = PlanSingleProblem(problem);

  EXPECT_EQ(LeafRegionOrder(plan.tree), std::vector<int>({0, 1}));
  EXPECT_EQ(plan.score.write_cut_cost, 0);
  EXPECT_EQ(plan.score.shared_read_cost, 0);
  EXPECT_EQ(plan.score.reorder_penalty, 0);

  ASSERT_EQ(plan.tree.size(), 1U);
  EXPECT_TRUE(plan.tree[0].is_scope);
  EXPECT_EQ(plan.tree[0].shell_axes, std::vector<std::string>({"i"}));
  const SunmmioTileLoopFusionPlannerTreeNode &inner_scope =
      ExpectSingleScopeChild(plan.tree[0]);
  EXPECT_EQ(inner_scope.shell_axes, std::vector<std::string>({"i", "j"}));
  ASSERT_EQ(inner_scope.children.size(), 2U);
  EXPECT_FALSE(inner_scope.children[0].is_scope);
  EXPECT_EQ(inner_scope.children[0].region_index, 0);
  EXPECT_FALSE(inner_scope.children[1].is_scope);
  EXPECT_EQ(inner_scope.children[1].region_index, 1);
}

TEST(SunmmioTileLoopFusionPlannerTest,
     ShallowDependenceSplitsOnlyInnerShellDespiteProfitableTileReuse) {
  Buffer row_buffer = MakeSharedBuffer("row_buffer", {I(8)});
  Buffer tile_buffer = MakeSharedBuffer("tile_buffer", {I(8), I(2)});

  SunmmioTileLoopFusionWindowProblem problem;
  problem.regions = {
      MakePlannerRegion(0, {"i", "j"}, MakeExtents({8, 2})),
      MakePlannerRegion(1, {"i", "j"}, MakeExtents({8, 2})),
  };
  problem.normalized_regions.resize(2);
  problem.normalized_regions[0].def_out = {
      MakeAccess(row_buffer, 1, 32),
      MakeAccess(tile_buffer, 2, 1024),
  };
  problem.normalized_regions[1].use_in = {
      MakeAccess(row_buffer, 1, 32),
      MakeAccess(tile_buffer, 2, 1024),
  };
  problem.graph.edges = {
      MakeRawEdge(0, 1, 0, 0, 1, 32, 1),
      MakeRawEdge(0, 1, 1, 1, 2, 1024, 2),
  };

  SunmmioTileLoopFusionWindowPlan plan = PlanSingleProblem(problem);

  EXPECT_EQ(LeafRegionOrder(plan.tree), std::vector<int>({0, 1}));
  ASSERT_EQ(plan.tree.size(), 1U);
  const SunmmioTileLoopFusionPlannerTreeNode &outer_scope = plan.tree[0];
  ASSERT_TRUE(outer_scope.is_scope);
  EXPECT_EQ(outer_scope.shell_axes, std::vector<std::string>({"i"}));
  ASSERT_EQ(outer_scope.children.size(), 2U);

  EXPECT_FALSE(outer_scope.children[0].is_scope);
  EXPECT_EQ(outer_scope.children[0].region_index, 0);
  EXPECT_FALSE(outer_scope.children[1].is_scope);
  EXPECT_EQ(outer_scope.children[1].region_index, 1);
  EXPECT_GT(plan.score.write_cut_cost, 0);
}

TEST(SunmmioTileLoopFusionPlannerTest,
     UnitTripInnerShellRemainsFusedAcrossShallowDependence) {
  Buffer row_buffer = MakeSharedBuffer("row_buffer", {I(8)});
  Buffer tile_buffer = MakeSharedBuffer("tile_buffer", {I(8), I(1)});

  SunmmioTileLoopFusionWindowProblem problem;
  problem.regions = {
      MakePlannerRegion(0, {"i", "j"}, MakeExtents({8, 1})),
      MakePlannerRegion(1, {"i", "j"}, MakeExtents({8, 1})),
  };
  problem.normalized_regions.resize(2);
  problem.normalized_regions[0].def_out = {
      MakeAccess(row_buffer, 1, 32),
      MakeAccess(tile_buffer, 2, 1024),
  };
  problem.normalized_regions[1].use_in = {
      MakeAccess(row_buffer, 1, 32),
      MakeAccess(tile_buffer, 2, 1024),
  };
  problem.graph.edges = {
      MakeRawEdge(0, 1, 0, 0, 1, 32, 1),
      MakeRawEdge(0, 1, 1, 1, 2, 1024, 2),
  };

  SunmmioTileLoopFusionWindowPlan plan = PlanSingleProblem(problem);

  ASSERT_EQ(plan.tree.size(), 1U);
  const SunmmioTileLoopFusionPlannerTreeNode &outer_scope = plan.tree[0];
  ASSERT_TRUE(outer_scope.is_scope);
  const SunmmioTileLoopFusionPlannerTreeNode &inner_scope =
      ExpectSingleScopeChild(outer_scope);
  EXPECT_EQ(inner_scope.shell_axes, std::vector<std::string>({"i", "j"}));
  ASSERT_EQ(inner_scope.children.size(), 2U);
  EXPECT_EQ(inner_scope.children[0].region_index, 0);
  EXPECT_EQ(inner_scope.children[1].region_index, 1);
}

TEST(SunmmioTileLoopFusionPlannerTest,
     LaterTileConsumerReordersButRank1ConsumerStaysOutsideRank2Shell) {
  Buffer row_buffer = MakeSharedBuffer("row_buffer", {I(8)});
  Buffer tile_buffer = MakeSharedBuffer("tile_buffer", {I(8), I(1)});

  SunmmioTileLoopFusionWindowProblem problem;
  problem.regions = {
      MakePlannerRegion(0, {"i", "j"}, MakeExtents({8, 1})),
      MakePlannerRegion(1, {"i"}, MakeExtents({8})),
      MakePlannerRegion(2, {"i", "j"}, MakeExtents({8, 1})),
  };
  problem.normalized_regions.resize(3);
  problem.normalized_regions[0].def_out = {
      MakeAccess(row_buffer, 1, 32),
      MakeAccess(tile_buffer, 2, 1024),
  };
  problem.normalized_regions[1].use_in = {MakeAccess(row_buffer, 1, 32)};
  problem.normalized_regions[2].use_in = {MakeAccess(tile_buffer, 2, 1024)};
  problem.graph.edges = {
      MakeRawEdge(0, 1, 0, 0, 1, 32),
      MakeRawEdge(0, 2, 1, 0, 2, 1024),
  };

  SunmmioTileLoopFusionWindowPlan plan = PlanSingleProblem(problem);

  EXPECT_EQ(LeafRegionOrder(plan.tree), std::vector<int>({0, 2, 1}));
  EXPECT_EQ(plan.score.write_cut_cost, 256);
  EXPECT_EQ(plan.score.shared_read_cost, 0);
  EXPECT_EQ(plan.score.reorder_penalty, 1);

  ASSERT_EQ(plan.tree.size(), 2U);
  EXPECT_TRUE(plan.tree[0].is_scope);
  EXPECT_EQ(plan.tree[0].shell_axes, std::vector<std::string>({"i"}));
  ASSERT_EQ(plan.tree[0].children.size(), 1U);
  EXPECT_TRUE(plan.tree[0].children[0].is_scope);
  EXPECT_FALSE(plan.tree[1].is_scope);
  EXPECT_EQ(plan.tree[1].region_index, 1);

  const SunmmioTileLoopFusionPlannerTreeNode &inner_scope =
      plan.tree[0].children[0];
  EXPECT_EQ(inner_scope.shell_axes, std::vector<std::string>({"i", "j"}));
  ASSERT_EQ(inner_scope.children.size(), 2U);
  EXPECT_FALSE(inner_scope.children[0].is_scope);
  EXPECT_EQ(inner_scope.children[0].region_index, 0);
  EXPECT_FALSE(inner_scope.children[1].is_scope);
  EXPECT_EQ(inner_scope.children[1].region_index, 2);
}

} // namespace
} // namespace tl
} // namespace tvm
