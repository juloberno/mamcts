// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef COST_CONSTRAINED_TEST_STATE_H
#define COST_CONSTRAINED_TEST_STATE_H

#include <iostream>
#include "mcts/state.h"

using namespace mcts;

// A simple environment with a 1D state, only if both agents select different actions, they get nearer to the terminal state
class CostConstrainedStatisticTestState : public mcts::StateInterface<CostConstrainedStatisticTestState>, 
  public RandomGenerator
{
public:
    CostConstrainedStatisticTestState(int n_steps, Cost collision_risk1, Cost collision_risk2,
                                     Reward reward_goal1, Reward reward_goal2, bool is_terminal) : 
                                     RandomGenerator(1000),
                                      current_state_(0), n_steps_(n_steps),
                                      collision_risk1_(collision_risk1), collision_risk2_(collision_risk2),
                                      reward_goal1_(reward_goal1), reward_goal2_(reward_goal2),
                                      is_terminal_(is_terminal) {}
    ~CostConstrainedStatisticTestState() {};

    std::shared_ptr<CostConstrainedStatisticTestState> clone() const {
        return std::make_shared<CostConstrainedStatisticTestState>(*this);
    }

    std::shared_ptr<CostConstrainedStatisticTestState> execute(const JointAction& joint_action, std::vector<Reward>& rewards, Cost& ego_cost) const {
        rewards.resize(1);
        rewards[0] = 0;

        if(joint_action == JointAction{0}) {
            return std::make_shared<CostConstrainedStatisticTestState>(*this);
        } else {
            bool is_terminal = false;
            bool collision = false;
            auto new_state_length = current_state_;
            std::uniform_real_distribution<> dist(0, 1);
            const auto sample = dist(random_generator_);
            const Probability to_goal_prob = get_transition_to_goal_probability(joint_action);
            if(joint_action == JointAction{1}) {
              if(sample <= to_goal_prob) {
                new_state_length += 1;
              } else {
                collision = true;
              }
            } else if(joint_action == JointAction{2}) {
              if(sample <= to_goal_prob) {
                new_state_length -= 1;
              } else {
                collision = true;
              }
            } else {
              throw std::logic_error("Invalid action selected");
            }
          if (current_state_ >= n_steps_) {
            rewards = std::vector<Reward>{reward_goal1_};
            ego_cost = 0.0f;
            is_terminal = true;
          } else if(current_state_ <= - n_steps_) {
            rewards = std::vector<Reward>{reward_goal2_};
            ego_cost = 0.0f;
            is_terminal = true;
          } else if (collision) {
            rewards = std::vector<Reward>{0.0f};
            ego_cost = 1.0f;
            is_terminal = true;
          }
          return std::make_shared<CostConstrainedStatisticTestState>(n_steps_, collision_risk1_, collision_risk2_,
                                                            reward_goal1_, reward_goal2_, is_terminal);
        }
    }

    Probability get_transition_to_goal_probability(const JointAction& joint_action) const {
      if(joint_action == JointAction{1}) {
            return std::pow(1 - collision_risk1_, n_steps_);
        } else if(joint_action == JointAction{2}) { 
            return std::pow(1 - collision_risk2_, n_steps_);
        } else {
          throw std::logic_error("Invalid action passed.");
        }
    }

    ActionIdx get_num_actions(AgentIdx agent_idx) const {
        return 3;
    }

    bool is_terminal() const {
        return is_terminal_;
    }

    const std::vector<AgentIdx> get_other_agent_idx() const {
        return std::vector<AgentIdx>{};
    }

    const AgentIdx get_ego_agent_idx() const {
        return 4;
    }

    std::string sprintf() const
    {
        std::stringstream ss;
        ss << "CostConstrainedStatisticTestState (current_state: " << current_state_ << ")";
        return ss.str();
    }
private:
    int current_state_;
    bool is_terminal_;

    // PARAMS
    const int n_steps_;
    const Reward reward_goal1_;
    const Reward reward_goal2_;
    const Cost collision_risk1_;
    const Cost collision_risk2_;
};



#endif 
