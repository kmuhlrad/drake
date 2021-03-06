/* clang-format off to disable clang-format-includes */
#include "drake/automotive/maliput/api/rules/road_rulebook.h"
/* clang-format on */
// TODO(maddog@tri.global) Satisfy clang-format via rules tests directory reorg.

#include <gtest/gtest.h>

#include "drake/automotive/maliput/api/rules/direction_usage_rule.h"
#include "drake/automotive/maliput/api/rules/regions.h"
#include "drake/automotive/maliput/api/rules/right_of_way_rule.h"
#include "drake/automotive/maliput/api/rules/speed_limit_rule.h"
#include "drake/common/drake_throw.h"

namespace drake {
namespace maliput {
namespace api {
namespace rules {
namespace {

// This class does not provide any semblance of useful functionality.
// It merely exercises the RoadRulebook abstract interface.
class MockRulebook : public RoadRulebook {
 public:
  const LaneSRange kZone{LaneId("some_lane"), {10., 20.}};
  const RightOfWayRule kRightOfWay{
    RightOfWayRule::Id("rowr_id"),
    LaneSRoute({kZone}), RightOfWayRule::ZoneType::kStopExcluded,
    {RightOfWayRule::State(
        RightOfWayRule::State::Id("green"),
        RightOfWayRule::State::Type::kGo,
        {})}};
  const SpeedLimitRule kSpeedLimit{SpeedLimitRule::Id("slr_id"),
                                   kZone,
                                   SpeedLimitRule::Severity::kStrict,
                                   0., 44.};
  const DirectionUsageRule kDirectionUsage{
    DirectionUsageRule::Id("dur_id"), kZone,
    {DirectionUsageRule::State(
      DirectionUsageRule::State::Id("dur_state"),
      DirectionUsageRule::State::Type::kWithS,
      DirectionUsageRule::State::Severity::kPreferred)}};

 private:
  virtual QueryResults DoFindRules(
      const std::vector<LaneSRange>& ranges, double) const {
    QueryResults results;
    if ((!ranges.empty()) &&
        (ranges[0].lane_id() == kZone.lane_id()) &&
        (ranges[0].s_range().s0() == kZone.s_range().s0()) &&
        (ranges[0].s_range().s1() == kZone.s_range().s1())) {
      results.right_of_way.push_back(kRightOfWay);
      results.speed_limit.push_back(kSpeedLimit);
      results.direction_usage.push_back(kDirectionUsage);
    }
    return results;
  }

  virtual RightOfWayRule DoGetRule(const RightOfWayRule::Id& id) const {
    if (id != kRightOfWay.id()) {
      throw std::out_of_range("");
    }
    return kRightOfWay;
  }

  virtual SpeedLimitRule DoGetRule(const SpeedLimitRule::Id& id) const {
    if (id != kSpeedLimit.id()) {
      throw std::out_of_range("");
    }
    return kSpeedLimit;
  }

  virtual DirectionUsageRule DoGetRule(const DirectionUsageRule::Id& id) const {
    if (id != kDirectionUsage.id()) {
      throw std::out_of_range("");
    }
    return kDirectionUsage;
  }
};


GTEST_TEST(RoadRulebookTest, ExerciseInterface) {
  const MockRulebook dut;

  const double kZeroTolerance = 0.;

  RoadRulebook::QueryResults nonempty = dut.FindRules({dut.kZone},
                                                      kZeroTolerance);
  EXPECT_EQ(nonempty.right_of_way.size(), 1);
  EXPECT_EQ(nonempty.speed_limit.size(), 1);
  EXPECT_EQ(nonempty.direction_usage.size(), 1);
  RoadRulebook::QueryResults empty = dut.FindRules({}, kZeroTolerance);
  EXPECT_EQ(empty.right_of_way.size(), 0);
  EXPECT_EQ(empty.speed_limit.size(), 0);
  EXPECT_EQ(empty.direction_usage.size(), 0);

  const double kNegativeTolerance = -1.;
  EXPECT_THROW(dut.FindRules({}, kNegativeTolerance),
               std::runtime_error);

  EXPECT_EQ(dut.GetRule(dut.kRightOfWay.id()).id(), dut.kRightOfWay.id());
  EXPECT_THROW(dut.GetRule(RightOfWayRule::Id("xxx")), std::out_of_range);

  EXPECT_EQ(dut.GetRule(dut.kSpeedLimit.id()).id(), dut.kSpeedLimit.id());
  EXPECT_THROW(dut.GetRule(SpeedLimitRule::Id("xxx")), std::out_of_range);

  EXPECT_EQ(dut.GetRule(dut.kDirectionUsage.id()).id(),
                        dut.kDirectionUsage.id());
  EXPECT_THROW(dut.GetRule(DirectionUsageRule::Id("xxx")),
                           std::out_of_range);
}


}  // namespace
}  // namespace rules
}  // namespace api
}  // namespace maliput
}  // namespace drake
