#include <gtest/gtest.h>

#include <limits>

#include "comm.h"
#include "sunmmio_utils.h"
#include "tileview/tileview_planner_common.h"

namespace tvm {
namespace tl {
namespace {

Target MakeA4ETarget() { return Target("llvm -mcpu=sunmmio-a4e"); }

struct DirectTransferCase {
  SunmmioTransferMechanism mechanism;
  const char *src_scope;
  const char *dst_scope;
  bool supported;
};

TEST(SunmmioDirectTransferTest, A4EScopeAndMechanismMatrix) {
  const DirectTransferCase cases[] = {
      {SunmmioTransferMechanism::kLocalDma, "global", kSunmmioScopeRSRAM, true},
      {SunmmioTransferMechanism::kLocalDma, "global", kSunmmioScopeWSRAM, true},
      {SunmmioTransferMechanism::kLocalDma, "global", kSunmmioScopeASRAM,
       false},
      {SunmmioTransferMechanism::kLocalDma, kSunmmioScopeRSRAM, "global", true},
      {SunmmioTransferMechanism::kLocalDma, kSunmmioScopeRSRAM,
       kSunmmioScopeASRAM, true},
      {SunmmioTransferMechanism::kLocalDma, kSunmmioScopeRSRAM,
       kSunmmioScopeWSRAM, true},
      {SunmmioTransferMechanism::kLocalDma, kSunmmioScopeRSRAM,
       kSunmmioScopeRSRAM, true},
      {SunmmioTransferMechanism::kTile, kSunmmioScopeRSRAM, kSunmmioScopeRSRAM,
       true},
      {SunmmioTransferMechanism::kTile, "global", kSunmmioScopeRSRAM, false},
      {SunmmioTransferMechanism::kHLink, kSunmmioScopeRSRAM, kSunmmioScopeASRAM,
       true},
      {SunmmioTransferMechanism::kHLink, kSunmmioScopeRSRAM, kSunmmioScopeRSRAM,
       true},
      {SunmmioTransferMechanism::kHLink, "global", kSunmmioScopeRSRAM, false},
      {SunmmioTransferMechanism::kHLink, kSunmmioScopeRSRAM, kSunmmioScopeWSRAM,
       false},
      {SunmmioTransferMechanism::kVLink, "global", kSunmmioScopeWSRAM, true},
      {SunmmioTransferMechanism::kVLink, "global", kSunmmioScopeRSRAM, true},
      {SunmmioTransferMechanism::kVLink, kSunmmioScopeRSRAM, kSunmmioScopeWSRAM,
       true},
      {SunmmioTransferMechanism::kVLink, kSunmmioScopeRSRAM, kSunmmioScopeRSRAM,
       true},
      {SunmmioTransferMechanism::kVLink, "global", kSunmmioScopeASRAM, false},
  };

  Target target = MakeA4ETarget();
  for (const DirectTransferCase &test_case : cases) {
    EXPECT_EQ(
        SupportsSunmmioDirectTransfer(target, test_case.mechanism,
                                      test_case.src_scope, DataType::Float(16),
                                      test_case.dst_scope, DataType::Float(16)),
        test_case.supported);
  }
}

TEST(SunmmioDirectTransferTest, A4EDTypePolicy) {
  Target target = MakeA4ETarget();

  EXPECT_FALSE(SupportsSunmmioDirectTransfer(
      target, SunmmioTransferMechanism::kLocalDma, kSunmmioScopeRSRAM,
      DataType::Float(32), "global", DataType::Float(16)));
  EXPECT_FALSE(SupportsSunmmioDirectTransfer(
      target, SunmmioTransferMechanism::kHLink, kSunmmioScopeRSRAM,
      DataType::Float(32), kSunmmioScopeRSRAM, DataType::Float(16)));
  EXPECT_FALSE(SupportsSunmmioDirectTransfer(
      target, SunmmioTransferMechanism::kVLink, kSunmmioScopeRSRAM,
      DataType::Float(32), kSunmmioScopeRSRAM, DataType::Float(16)));
  EXPECT_TRUE(SupportsSunmmioDirectTransfer(
      target, SunmmioTransferMechanism::kTile, kSunmmioScopeRSRAM,
      DataType::Float(32), kSunmmioScopeRSRAM, DataType::Float(16)));
}

TEST(SunmmioDirectTransferTest, A4EDirectCopyUsesSupportedMechanism) {
  Target target = MakeA4ETarget();

  EXPECT_TRUE(SupportsSunmmioDirectCopy(target, "global", DataType::Float(16),
                                        kSunmmioScopeRSRAM,
                                        DataType::Float(16)));
  EXPECT_TRUE(SupportsSunmmioDirectCopy(target, kSunmmioScopeRSRAM,
                                        DataType::Float(32), kSunmmioScopeRSRAM,
                                        DataType::Float(16)));
  EXPECT_FALSE(SupportsSunmmioDirectCopy(target, "global", DataType::Float(32),
                                         kSunmmioScopeRSRAM,
                                         DataType::Float(16)));
  EXPECT_FALSE(
      SupportsSunmmioDirectCopy(target, kSunmmioScopeASRAM, DataType::Float(16),
                                kSunmmioScopeRSRAM, DataType::Float(16)));
}

TEST(SunmmioDirectTransferTest, A4EMapsCommunicationDirectionsToLinks) {
  Target target = MakeA4ETarget();

  EXPECT_FALSE(SupportsSunmmioDirectCommunication(
      target, CommunicationDirections::kHorizontal, "global",
      DataType::Float(16), kSunmmioScopeRSRAM, DataType::Float(16)));
  EXPECT_TRUE(SupportsSunmmioDirectCommunication(
      target, CommunicationDirections::kVertical, "global", DataType::Float(16),
      kSunmmioScopeRSRAM, DataType::Float(16)));
  EXPECT_FALSE(SupportsSunmmioDirectCommunication(
      target, CommunicationDirections::kHorizontalAndVertical, "global",
      DataType::Float(16), kSunmmioScopeRSRAM, DataType::Float(16)));
  EXPECT_TRUE(SupportsSunmmioDirectCommunication(
      target, CommunicationDirections::kHorizontalAndVertical,
      kSunmmioScopeRSRAM, DataType::Float(16), kSunmmioScopeRSRAM,
      DataType::Float(16)));
}

TEST(SunmmioTileChunkAlignmentTest, LargeStaticExtentsDoNotOverflow) {
  constexpr int kMaxExtent = std::numeric_limits<int>::max();

  EXPECT_FALSE(IsTileChunkRsramAligned(kMaxExtent, kMaxExtent, 32, 64));
  EXPECT_TRUE(IsTileChunkRsramAligned(kMaxExtent, kMaxExtent - 15, 32, 64));
  EXPECT_TRUE(IsTileChunkRsramAligned(kMaxExtent, kMaxExtent - 7, 32, 96));
}

} // namespace
} // namespace tl
} // namespace tvm
