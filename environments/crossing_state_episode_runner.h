// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================


#ifndef MCTS_EPISODE_RUNNER_H_
#define MCTS_EPISODE_RUNNER_H_

#include "environments/crossing_state.h"
#include "environments/viewer.h"

namespace mcts {

class CrossingStateEpisodeRunner {
  public:
    CrossingStateEpisodeRunner(const std::unordered_map<AgentIdx, AgentPolicyCrossingState>& agents_true_policies,
                              const std::vector<AgentPolicyCrossingState>& hypothesis,
                              const unsigned int max_steps,
                              Viewer* viewer) :
                  agents_true_policies_(agents_true_policies),
                  current_state_(),
                  last_state_(),
                  belief_tracker_(4,1, HypothesisBeliefTracker::PRODUCT),
                  MAX_STEPS(max_steps),
                  current_step_(0),
                  viewer_(viewer)  {
                  RandomGenerator::random_generator_ = std::mt19937(1000);
                  current_state_ = std::make_shared<CrossingState>(belief_tracker_.sample_current_hypothesis());
                  for(const auto& hp : hypothesis) {
                    current_state_->add_hypothesis(hp);
                  }
                  last_state_ = current_state_;
                  // Init tracking
                  belief_tracker_.belief_update(*last_state_, *current_state_);
                  }


    std::tuple<float, float, bool, bool, bool> step() {
      std::vector<Reward> rewards;
      Cost cost;

      JointAction jointaction(current_state_->get_agent_idx().size());
      for (auto agent_idx : current_state_->get_agent_idx()) {
        if (agent_idx == CrossingState::ego_agent_idx ) {
          // Plan for ego agent with hypothesis-based search
          Mcts<CrossingState, UctStatistic, HypothesisStatistic, RandomHeuristic> mcts;
          mcts.search(*current_state_, belief_tracker_, 5000, 10000);
          jointaction[agent_idx] = mcts.returnBestAction();
          std::cout << "best uct action: " << idx_to_ego_crossing_action(jointaction[agent_idx]) << std::endl;
        } else {
          // Other agents act according to unknown true agents policy
          const auto action = agents_true_policies_.at(agent_idx).act(current_state_->get_agent_state(agent_idx),
                                                      current_state_->get_ego_state().x_pos);
          jointaction[agent_idx] = aconv(action);
        }
      }
      std::cout << "Step " << i << ", Action = " << jointaction << ", " << current_state_->sprintf() << std::endl;
      last_state_ = current_state_;
      current_state_ = current_state_->execute(jointaction, rewards, cost);
      belief_tracker_.belief_update(*last_state_, *current_state_);
      
      const bool collision = current_state_->is_terminal() && !current_state_->ego_goal_reached();
      const bool goal_reached = current_state_->ego_goal_reached();
      current_step_ += 1;
      const bool max_steps = current_step_ > MAX_STEPS;

      if(viewer_) {
        current_state_->draw(viewer_);
      }

      return std::make_tuple<float, float, bool, bool, bool> (rewards[CrossingState::ego_agent_idx], 
                                                              cost,
                                                              collision,
                                                              goal_reached,
                                                              max_steps);
    }

  private:
    Viewer* viewer_;
    std::shared_ptr<CrossingState> current_state_;
    std::shared_ptr<CrossingState> last_state_;
    HypothesisBeliefTracker belief_tracker_; // todo: pass params
    const std::unordered_map<AgentIdx,AgentPolicyCrossingState> agents_true_policies_;
    const unsigned int MAX_STEPS;
    unsigned int current_step_;
};


} // namespace mcts

#endif // MCTS_EPISODE_RUNNER_H_