/**
 * @file This file implements a state machine that drives the kuka iiwa arm to
 * pick up a block from one table to place it on another repeatedly.
 */

#include <iostream>
#include <list>
#include <memory>

#include <lcm/lcm-cpp.hpp>
#include "bot_core/robot_state_t.hpp"

#include "drake/common/drake_assert.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/find_resource.h"
#include "drake/examples/kuka_iiwa_arm/dev/explore_object/state_machine_system.h"
#include "drake/examples/kuka_iiwa_arm/pick_and_place/world_state.h"
#include "drake/lcmt_iiwa_status.hpp"
#include "drake/lcmt_schunk_wsg_command.hpp"
#include "drake/lcmt_schunk_wsg_status.hpp"
#include "drake/manipulation/planner/constraint_relaxing_ik.h"
#include "drake/util/lcmUtil.h"

namespace drake {
namespace examples {
namespace kuka_iiwa_arm {
namespace explore_object {
namespace {
using manipulation::planner::ConstraintRelaxingIk;
using pick_and_place::WorldState;

class WorldStateSubscriber {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(WorldStateSubscriber)

  WorldStateSubscriber(lcm::LCM* lcm, WorldState* state)
      : lcm_(lcm),
        state_(state) {
    DRAKE_DEMAND(state);

    lcm_subscriptions_.push_back(
        lcm_->subscribe("EST_ROBOT_STATE",
                        &WorldStateSubscriber::HandleIiwaStatus, this));
    lcm_subscriptions_.push_back(
        lcm_->subscribe("SCHUNK_WSG_STATUS",
                        &WorldStateSubscriber::HandleWsgStatus, this));
    lcm_subscriptions_.push_back(
        lcm_->subscribe("OBJECT_STATE_EST",
                        &WorldStateSubscriber::HandleObjectStatus, this));
  }

  ~WorldStateSubscriber() {
    for (lcm::Subscription* sub : lcm_subscriptions_) {
      int status = lcm_->unsubscribe(sub);
      DRAKE_DEMAND(status == 0);
    }
    lcm_subscriptions_.clear();
  }

 private:
  // Handles iiwa states from the LCM message.
  void HandleIiwaStatus(const lcm::ReceiveBuffer*, const std::string&,
                        const bot_core::robot_state_t* iiwa_msg) {
    DRAKE_DEMAND(iiwa_msg != nullptr);
    state_->HandleIiwaStatus(*iiwa_msg);
  }

  // Handles WSG states from the LCM message.
  void HandleWsgStatus(const lcm::ReceiveBuffer*, const std::string&,
                       const lcmt_schunk_wsg_status* wsg_msg) {
    DRAKE_DEMAND(wsg_msg != nullptr);
    state_->HandleWsgStatus(*wsg_msg);
  }

  // Handles object states from the LCM message.
  void HandleObjectStatus(const lcm::ReceiveBuffer*,
                          const std::string&,
                          const bot_core::robot_state_t* obj_msg) {
    DRAKE_DEMAND(obj_msg != nullptr);
    state_->HandleObjectStatus(*obj_msg);
  }

  // LCM subscription management.
  lcm::LCM* lcm_;
  WorldState* state_;
  std::list<lcm::Subscription*> lcm_subscriptions_;
};


// Makes a state machine that drives the iiwa to pick up a block from one table
// and place it on the other table.
void RunExploreObjectDiscreteDemo() {
  lcm::LCM lcm;

  const std::string iiwa_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/urdf/"
      "iiwa14_primitive_collision.urdf");
  const std::string iiwa_end_effector_name = "iiwa_link_ee";

  // Makes a WorldState, and sets up LCM subscriptions.
  WorldState env_state(iiwa_path, iiwa_end_effector_name);
  WorldStateSubscriber env_state_subscriber(&lcm, &env_state);

  // Spins until at least one message is received from every LCM channel.
  while (lcm.handleTimeout(10) == 0 || env_state.get_iiwa_time() == -1 ||
         env_state.get_obj_time() == -1 || env_state.get_wsg_time() == -1) {
  }

  // TODO: Change to ExploreObjectStateMachine, make sure the methods are still
  // relevant and will work with the planner.
  // Makes a planner.
  const Isometry3<double>& iiwa_base = env_state.get_iiwa_base();
  ConstraintRelaxingIk planner(
      iiwa_path, iiwa_end_effector_name, iiwa_base);
  ExploreObjectStateMachine::IiwaPublishCallback iiwa_callback =
      ([&](const robotlocomotion::robot_plan_t* plan) {
        lcm.publish("COMMITTED_ROBOT_PLAN", plan);
      });

  ExploreObjectStateMachine::WsgPublishCallback wsg_callback =
      ([&](const lcmt_schunk_wsg_command* msg) {
        std::cout << "publishing wsg command" << std::endl;
        lcm.publish("SCHUNK_WSG_COMMAND", msg);
      });

  // This needs to match the object model file in iiwa_wsg_simulation.cc
  Eigen::Vector3d half_box_height(0, 0, 0.1);

  // TODO(kmuhlrad): all of these locations need to be updated
  std::vector<Eigen::Vector3d> absolute_grasp_locations;
  // Eigen::Vector3d(0.86, -0.36, 0.13), 0.09, 0.05, 0.01
  absolute_grasp_locations.push_back(Eigen::Vector3d(0.80, -0.03, 0.13));
  absolute_grasp_locations.push_back(Eigen::Vector3d(0.80, -0.03, 0.09));
  absolute_grasp_locations.push_back(Eigen::Vector3d(0.80, -0.03, 0.05));
  absolute_grasp_locations.push_back(Eigen::Vector3d(0.80, -0.03, 0.01));

  // Define the transformation to where the object will be sitting, relative to
  // the robot base.
  std::vector<Isometry3<double>> relative_grasp_locations;
  Isometry3<double> relative_grasp_location;

  for (Eigen::Vector3d& absolute_grasp_loc: absolute_grasp_locations) {
      relative_grasp_location.translation() = absolute_grasp_loc + half_box_height;
      relative_grasp_location.linear().setIdentity();
      relative_grasp_location.rotate(Eigen::AngleAxisd(0.25*M_PI, Eigen::Vector3d::UnitZ()));
      relative_grasp_locations.push_back(relative_grasp_location);
  }

  /*
  std::vector<Isometry3<double>> place_locations;
  Isometry3<double> place_location;
  // TODO(sam.creasey) fix these
  place_location.translation() = Vector3<double>(0, 0.8, half_box_height);
  place_location.linear() = Matrix3<double>(
      AngleAxis<double>(M_PI / 2., Vector3<double>::UnitZ()));
  place_locations.push_back(place_location);

  place_location.translation() = Vector3<double>(0.8, 0, half_box_height);
  place_location.linear().setIdentity();
  place_locations.push_back(place_location);
  */

  ExploreObjectStateMachine machine(relative_grasp_locations, true);

  // lcm handle loop
  while (true) {
    // Handles all messages.
    while (lcm.handleTimeout(10) == 0) {
      // drake::log()->info("Timing out");
    }
    machine.Update(env_state, iiwa_callback, wsg_callback, &planner);
  }
}

}  // namespace
}  // namespace explore_object
}  // namespace kuka_iiwa_arm
}  // namespace examples
}  // namespace drake

int main() {
  drake::examples::kuka_iiwa_arm::explore_object::RunExploreObjectDiscreteDemo();
  return 0;
}
