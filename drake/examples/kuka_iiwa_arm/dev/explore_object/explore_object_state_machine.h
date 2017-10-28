#ifndef DRAKE_EXAMPLES_KUKA_IIWA_ARM_DEV_EXPLORE_OBJECTS_EXPLORE_OBJECTS_STATE_MACHINE_H
#define DRAKE_EXAMPLES_KUKA_IIWA_ARM_DEV_EXPLORE_OBJECTS_EXPLORE_OBJECTS_STATE_MACHINE_H

#include <functional>
#include <memory>
#include <vector>

#include "robotlocomotion/robot_plan_t.hpp"

#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/examples/kuka_iiwa_arm/pick_and_place/action.h"
#include "drake/examples/kuka_iiwa_arm/pick_and_place/world_state.h"
#include "drake/lcmt_schunk_wsg_command.hpp"
#include "drake/manipulation/planner/constraint_relaxing_ik.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace explore_object {

/// Different states for the pick and place task.
enum ExploreObjectState {
  kOpenGripper,
  kApproachPregrasp,
  kApproachGrasp,
  kGrasp,
  kDone,
};

/// A class which controls the actions for touching many places on a
/// single target in the environment.
class ExploreObjectStateMachine {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ExploreObjectStateMachine)

  typedef std::function<void(
      const robotlocomotion::robot_plan_t*)> IiwaPublishCallback;
  typedef std::function<void(
      const lcmt_schunk_wsg_command*)> WsgPublishCallback;

  /// Construct an explore objects state machine.  @p grasp_locations
  /// should contain a list of locations to grasp the object.  The
  /// state machine will cycle through the grasp locations, grasping 
  /// the item then releasing it, then grasping again at the next location.
  /// If @p loop is true, the state machine will loop through the grasp
  /// locations, otherwise it will remain in the kDone state once
  /// complete.
  ExploreObjectStateMachine(
      const std::vector<Isometry3<double>>& grasp_locations, bool loop);
  ~ExploreObjectStateMachine();

  /// Update the state machine based on the state of the world in @p
  /// env_state.  When a new robot plan is available, @p iiwa_callback
  /// will be invoked with the new plan.  If the desired gripper state
  /// changes, @p wsg_callback is invoked.  @p planner should contain
  /// an appropriate planner for the robot.
  void Update(const pick_and_place::WorldState& env_state,
              const IiwaPublishCallback& iiwa_callback,
              const WsgPublishCallback& wsg_callback,
              manipulation::planner::ConstraintRelaxingIk* planner);

  ExploreObjectState state() const { return state_; }

 private:
  std::vector<Isometry3<double>> grasp_locations_;
  int next_grasp_location_;
  bool loop_;

  pick_and_place::WsgAction wsg_act_;
  pick_and_place::IiwaMove iiwa_move_;

  ExploreObjectState state_;

  // Poses used for storing end-points of Iiwa trajectories at various states
  // of the demo.
  Isometry3<double> X_Wend_effector_0_;
  Isometry3<double> X_Wend_effector_1_;

  // Desired object end pose relative to the base of the iiwa arm.
  Isometry3<double> X_IIWAobj_desired_;

  // Desired object end pose in the world frame.
  Isometry3<double> X_Wobj_desired_;

  Vector3<double> tight_pos_tol_;
  double tight_rot_tol_;

  Vector3<double> loose_pos_tol_;
  double loose_rot_tol_;
};

}  // namespace explore_object
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

#endif  // DRAKE_EXAMPLES_KUKA_IIWA_ARM_DEV_EXPLORE_OBJECTS_EXPLORE_OBJECTS_STATE_MACHINE_H