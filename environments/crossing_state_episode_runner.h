// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================


#ifndef MCTS_EPISODE_RUNNER_H_
#define MCTS_EPISODE_RUNNER_H_

#include "environments/crossing_state.h"
#include "mcts/heuristics/random_heuristic.h"
#include "mcts/statistics/uct_statistic.h"
#include "mcts/hypothesis/hypothesis_statistic.h"
#include "mcts/hypothesis/hypothesis_belief_tracker.h"
#include "environments/viewer.h"

namespace mcts {


class CrossingStateEpisodeRunner {
  public:
    CrossingStateEpisodeRunner(const std::unordered_map<AgentIdx, AgentPolicyCrossingState>& agents_true_policies,
                              const std::vector<AgentPolicyCrossingState>& hypothesis,
                              const unsigned int& max_steps,
                              const unsigned int& belief_tracking_hist_len,
                              const float& belief_tracking_discount,
                              const HypothesisBeliefTracker::PosteriorType& posterior_type,
                              const unsigned int& mcts_max_search_time,
                              const unsigned int& mcts_max_iterations,
                              Viewer* viewer) :
                  agents_true_policies_(agents_true_policies),
                  current_state_(),
                  last_state_(),
                  belief_tracker_(belief_tracking_hist_len,belief_tracking_discount, posterior_type),
                  max_steps_(max_steps),
                  mcts_max_search_time_(mcts_max_search_time),
                  mcts_max_iterations_(mcts_max_iterations),
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

    // Reward, Cost, Terminal, Collision, GoalReached, MaxSteps
    typedef std::tuple<float, float, bool, bool, bool, bool> StepResult;
    StepResult step() {
      if(current_state_->is_terminal()) {
        std::cout << "Step " << current_step_ << "!!! terminal state reached  " << current_state_->sprintf() << std::endl;
        return std::tuple<float, float, bool, bool, bool, bool>();
      }
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
      std::cout << "Step " << current_step_ << ", Action = " << jointaction << ", " << current_state_->sprintf() << std::endl;
      last_state_ = current_state_;
      current_state_ = last_state_->execute(jointaction, rewards, cost);
      belief_tracker_.belief_update(*last_state_, *current_state_);
      
      bool collision = current_state_->is_terminal() && !current_state_->ego_goal_reached();
      bool goal_reached = current_state_->ego_goal_reached();
      current_step_ += 1;
      bool max_steps = current_step_ > max_steps_;

      if(viewer_) {
        current_state_->draw(viewer_);
      }

      return std::make_tuple<float, float,bool, bool, bool, bool> (rewards[CrossingState::ego_agent_idx], 
                                                              cost,
                                                              std::move(current_state_->is_terminal()),
                                                              std::move(collision),
                                                              std::move(goal_reached),
                                                              std::move(max_steps));
    }

    static const std::vector<std::string> EVAL_RESULT_COLUMN_DESC;
    typedef std::vector<StepResult> EpisodeResult;
    EpisodeResult run() {
      EpisodeResult episode_result;
      bool done = false;
      while(!done) {
        const auto step_result = step();
        episode_result.push_back(step_result);
        if(std::get<2>(step_result) || std::get<5>(step_result) ) {
          break;
        }
      }
    return std::move(episode_result);
    }

  private:
    Viewer* viewer_;
    std::shared_ptr<CrossingState> current_state_;
    std::shared_ptr<CrossingState> last_state_;
    HypothesisBeliefTracker belief_tracker_; // todo: pass params
    const std::unordered_map<AgentIdx, AgentPolicyCrossingState> agents_true_policies_;
    const unsigned int max_steps_;
    const unsigned int mcts_max_search_time_;
    const unsigned int mcts_max_iterations_;
    unsigned int current_step_;
};


} // namespace mcts

#endif // MCTS_EPISODE_RUNNER_H_