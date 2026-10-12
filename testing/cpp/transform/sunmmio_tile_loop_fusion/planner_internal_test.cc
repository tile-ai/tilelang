#include <gtest/gtest.h>

#include <tvm/runtime/logging.h>

#include "transform/sunmmio_tile_loop_fusion/planner_internal.h"

namespace tvm {
namespace tl {
namespace planner_internal {
namespace {

ResidentValueState MakeResident(int64_t payload_bytes, int64_t instance_count) {
  ResidentValueState resident;
  resident.kind = ResidentValueKind::kDefinition;
  resident.origin_region_local_index = 3;
  resident.buffer_region_id = 7;
  resident.buffer_name = "debug_buffer";
  resident.home_depth = 2;
  resident.payload_bytes = payload_bytes;
  resident.instance_count = instance_count;
  return resident;
}

Array<PrimExpr> MakeUnitExtentPrefix(int depth) {
  Array<PrimExpr> extents;
  for (int i = 0; i < depth; ++i) {
    extents.push_back(IntImm(DataType::Int(32), 1));
  }
  return extents;
}

WindowPlannerInput MakeRawCoverageInput() {
  WindowPlannerInput input;
  auto *problem = new SunmmioTileLoopFusionWindowProblem();
  problem->regions.resize(3);

  input.problem = problem;
  input.regions.resize(3);
  input.edges.resize(2);
  input.incoming_edges_by_dst.resize(3);
  input.outgoing_edges_by_src.resize(3);
  input.predecessor_masks.assign(3, DynamicBitset(3));
  input.earlier_source_masks.assign(3, DynamicBitset(3));

  for (int i = 0; i < 3; ++i) {
    problem->regions[i].global_region_index = i;
    problem->regions[i].logical_execution_axis_keys = {"i"};
    problem->regions[i].execution_loop_extents = MakeUnitExtentPrefix(1);
    input.regions[i].global_region_index = i;
  }

  input.regions[1].use_in.push_back({10, "debug_buffer", 1, 64, 1});
  input.regions[1].use_in.push_back({11, "debug_buffer", 1, 80, 1});
  input.regions[2].use_in.push_back({10, "debug_buffer", 1, 64, 1});

  WindowPlannerEdgeInfo first_edge;
  first_edge.src_local_index = 0;
  first_edge.dst_local_index = 1;
  first_edge.kind = TileScopeDependenceKind::kRAW;
  first_edge.buffer_region_id = 10;
  first_edge.buffer_name = "debug_buffer";
  first_edge.rho = 1;
  first_edge.max_shared_execution_depth = 1;
  first_edge.weight = 64;
  first_edge.instance_count = 1;
  first_edge.covered_use_index = 0;
  input.edges[0] = first_edge;

  WindowPlannerEdgeInfo second_edge = first_edge;
  second_edge.dst_local_index = 2;
  input.edges[1] = second_edge;

  input.incoming_edges_by_dst[1].push_back(0);
  input.incoming_edges_by_dst[2].push_back(1);
  input.outgoing_edges_by_src[0].push_back(0);
  input.outgoing_edges_by_src[0].push_back(1);
  SetBit(&input.predecessor_masks[1], 0);
  SetBit(&input.predecessor_masks[2], 0);
  SetBit(&input.earlier_source_masks[1], 0);
  SetBit(&input.earlier_source_masks[2], 0);
  SetBit(&input.earlier_source_masks[2], 1);

  DynamicBitset raw_consumers(3);
  SetBit(&raw_consumers, 1);
  SetBit(&raw_consumers, 2);
  input.raw_consumer_masks_by_key.emplace(RawConsumerKey{0, 10}, raw_consumers);

  return input;
}

TEST(SunmmioTileLoopFusionPlannerInternalTest,
     DynamicBitsetRejectsPaddingBits) {
  DynamicBitset bitset(19);
  SetBit(&bitset, 18);

  EXPECT_TRUE(TestBit(bitset, 18));
  EXPECT_THROW(TestBit(bitset, 19), tvm::runtime::InternalError);
}

TEST(SunmmioTileLoopFusionPlannerInternalTest,
     InstallResidentIfMissingDedupesByFullIdentity) {
  std::vector<OpenScopeFrame> identical_scopes(2);
  ResidentValueState base = MakeResident(64, 8);
  InstallResidentIfMissing(&identical_scopes, base);
  InstallResidentIfMissing(&identical_scopes, base);
  ASSERT_EQ(identical_scopes[1].residents.size(), 1U);
  EXPECT_TRUE(SameResidentValue(identical_scopes[1].residents[0], base));

  std::vector<OpenScopeFrame> payload_scopes(2);
  InstallResidentIfMissing(&payload_scopes, base);
  InstallResidentIfMissing(&payload_scopes, MakeResident(128, 8));
  EXPECT_EQ(payload_scopes[1].residents.size(), 2U);

  std::vector<OpenScopeFrame> instance_scopes(2);
  InstallResidentIfMissing(&instance_scopes, base);
  InstallResidentIfMissing(&instance_scopes, MakeResident(64, 16));
  EXPECT_EQ(instance_scopes[1].residents.size(), 2U);
}

TEST(SunmmioTileLoopFusionPlannerInternalTest,
     PlannerStateSerializationIncludesPendingDependences) {
  PlannerState first{DynamicBitset(2), std::vector<OpenScopeFrame>(2)};
  PlannerState second = first;
  first.open_scopes[1].pending_edge_indices = {1};
  second.open_scopes[1].pending_edge_indices = {2};

  EXPECT_NE(SerializePlannerState(first), SerializePlannerState(second));
}

TEST(SunmmioTileLoopFusionPlannerInternalTest,
     PlannerStateSerializationIncludesExecutionRank) {
  PlannerState first{DynamicBitset(2), std::vector<OpenScopeFrame>(1)};
  first.open_scopes[0].shell_axes = {"i"};
  first.open_scopes[0].shell_extents = MakeUnitExtentPrefix(1);
  first.open_scopes[0].execution_rank = 1;
  PlannerState second = first;
  second.open_scopes[0].execution_rank = 2;

  EXPECT_NE(SerializePlannerState(first), SerializePlannerState(second));
}

TEST(SunmmioTileLoopFusionPlannerInternalTest,
     ApplyActionTracksPendingDependences) {
  for (TileScopeDependenceKind kind :
       {TileScopeDependenceKind::kRAW, TileScopeDependenceKind::kWAR,
        TileScopeDependenceKind::kWAW}) {
    WindowPlannerInput input;
    auto *problem = new SunmmioTileLoopFusionWindowProblem();
    problem->regions.resize(2);
    for (int i = 0; i < 2; ++i) {
      problem->regions[i].global_region_index = i;
      problem->regions[i].logical_execution_axis_keys = {"i", "j"};
      problem->regions[i].execution_loop_extents = {
          IntImm(DataType::Int(32), 8), IntImm(DataType::Int(32), 2)};
    }

    input.problem = problem;
    input.regions.resize(2);
    input.incoming_edges_by_dst.resize(2);
    input.outgoing_edges_by_src.resize(2);
    input.predecessor_masks.assign(2, DynamicBitset(2));
    input.earlier_source_masks.assign(2, DynamicBitset(2));

    WindowPlannerEdgeInfo edge;
    edge.src_local_index = 0;
    edge.dst_local_index = 1;
    edge.kind = kind;
    edge.max_shared_execution_depth = 1;
    input.edges.push_back(edge);
    input.outgoing_edges_by_src[0].push_back(0);
    input.incoming_edges_by_dst[1].push_back(0);

    PlannerState initial{DynamicBitset(2), {}};
    TransitionResult transition = ApplyAction(input, initial, 0, 0, 2);

    ASSERT_EQ(transition.next_state.open_scopes.size(), 2U);
    EXPECT_EQ(transition.next_state.open_scopes[1].pending_edge_indices,
              std::vector<int>({0}));
    delete problem;
  }
}

TEST(SunmmioTileLoopFusionPlannerInternalTest,
     RawCoverageOnlySuppressesExactCoveredUses) {
  WindowPlannerInput input = MakeRawCoverageInput();
  PlannerState state{DynamicBitset(3), {}};

  TransitionResult first = ApplyAction(input, state, 1, 0, 1);
  TransitionResult second = ApplyAction(input, first.next_state, 2, 1, 1);

  EXPECT_EQ(first.delta.write_cut_cost, 64);
  EXPECT_EQ(first.delta.shared_read_cost, 80);
  EXPECT_EQ(second.delta.write_cut_cost, 0);
  EXPECT_EQ(second.delta.shared_read_cost, 0);

  delete input.problem;
}

WindowPlannerInput MakeTwoRegionSolverInput() {
  WindowPlannerInput input;
  auto *problem = new SunmmioTileLoopFusionWindowProblem();
  problem->regions.resize(2);

  input.problem = problem;
  input.regions.resize(2);
  input.incoming_edges_by_dst.resize(2);
  input.outgoing_edges_by_src.resize(2);
  input.predecessor_masks.assign(2, DynamicBitset(2));
  input.earlier_source_masks.assign(2, DynamicBitset(2));

  for (int i = 0; i < 2; ++i) {
    problem->regions[i].global_region_index = i;
    problem->regions[i].logical_execution_axis_keys = {"i"};
    problem->regions[i].execution_loop_extents = MakeUnitExtentPrefix(1);
    input.regions[i].global_region_index = i;
  }

  SetBit(&input.earlier_source_masks[1], 0);
  return input;
}

TEST(SunmmioTileLoopFusionPlannerInternalTest,
     SolveWindowPlanPrefersSourceOrderWhenCostsAreOtherwiseEqual) {
  WindowPlannerInput input = MakeTwoRegionSolverInput();
  PlannerState state{DynamicBitset(2), {}};
  PlannerSearchContext context;

  MemoResult result = SolveWindowPlan(input, state, &context);

  EXPECT_EQ(result.score.write_cut_cost, 0);
  EXPECT_EQ(result.score.shared_read_cost, 0);
  EXPECT_EQ(result.score.live_range_penalty, 0);
  EXPECT_EQ(result.score.reorder_penalty, 0);
  ASSERT_EQ(result.actions.size(), 2U);
  EXPECT_EQ(result.actions[0].region_index, 0);
  EXPECT_EQ(result.actions[1].region_index, 1);
  EXPECT_EQ(result.actions[0].close_to_depth, 0);
  EXPECT_EQ(result.actions[0].open_to_depth, 0);
  EXPECT_EQ(result.actions[1].close_to_depth, 0);
  EXPECT_EQ(result.actions[1].open_to_depth, 0);

  delete input.problem;
}

TEST(SunmmioTileLoopFusionPlannerInternalTest,
     BuildPlanTreeReconstructsNestedShellStructure) {
  SunmmioTileLoopFusionPlannerAction first;
  first.region_index = 0;
  first.close_to_depth = 0;
  first.open_to_depth = 1;
  first.opened_shells = {{"i"}};
  first.opened_shell_extents = {MakeUnitExtentPrefix(1)};

  SunmmioTileLoopFusionPlannerAction second;
  second.region_index = 1;
  second.close_to_depth = 1;
  second.open_to_depth = 2;
  second.opened_shells = {{"i", "j"}};
  second.opened_shell_extents = {MakeUnitExtentPrefix(2)};

  SunmmioTileLoopFusionPlannerAction third;
  third.region_index = 2;
  third.close_to_depth = 1;
  third.open_to_depth = 1;

  std::vector<SunmmioTileLoopFusionPlannerAction> actions{first, second, third};
  std::vector<SunmmioTileLoopFusionPlannerTreeNode> tree =
      BuildPlanTree(actions);

  ASSERT_EQ(tree.size(), 1U);
  EXPECT_TRUE(tree[0].is_scope);
  EXPECT_EQ(tree[0].shell_axes, std::vector<std::string>({"i"}));
  ASSERT_EQ(tree[0].children.size(), 3U);

  EXPECT_FALSE(tree[0].children[0].is_scope);
  EXPECT_EQ(tree[0].children[0].region_index, 0);

  EXPECT_TRUE(tree[0].children[1].is_scope);
  EXPECT_EQ(tree[0].children[1].shell_axes,
            std::vector<std::string>({"i", "j"}));
  ASSERT_EQ(tree[0].children[1].children.size(), 1U);
  EXPECT_FALSE(tree[0].children[1].children[0].is_scope);
  EXPECT_EQ(tree[0].children[1].children[0].region_index, 1);

  EXPECT_FALSE(tree[0].children[2].is_scope);
  EXPECT_EQ(tree[0].children[2].region_index, 2);
}

} // namespace
} // namespace planner_internal
} // namespace tl
} // namespace tvm
