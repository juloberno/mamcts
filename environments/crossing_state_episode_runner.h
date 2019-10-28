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

template<typename Domain>
class CrossingStateEpisodeRunner {
  public:
    CrossingStateEpisodeRunner(const std::unordered_map<AgentIdx, AgentPolicyCrossingState<Domain>>& agents_true_policies,
                              const std::vector<AgentPolicyCrossingState<Domain>>& hypothesis,
                              const unsigned int& max_steps,
                              const unsigned int& belief_tracking_hist_len,
                              const float& belief_tracking_discount,
                              const HypothesisBeliefTracker::PosteriorType& posterior_type,
                              const unsigned int& mcts_max_search_time,
                              const unsigned int& mcts_max_iterations,
                              const MctsParameters& mcts_parameters,
                              Viewer* viewer) :
                  agents_true_policies_(agents_true_policies),
                  current_state_(),
                  last_state_(),
                  belief_tracker_(belief_tracking_hist_len, belief_tracking_discount, posterior_type),
                  max_steps_(max_steps),
                  mcts_max_search_time_(mcts_max_search_time),
                  mcts_max_iterations_(mcts_max_iterations),
                  mcts_parameters_(mcts_parameters),
                  viewer_(viewer)  {
                  RandomGenerator::random_generator_ = std::mt19937(1000);
                  current_state_ = std::make_shared<CrossingState<Domain>>(belief_tracker_.sample_current_hypothesis());
                  for(const auto& hp : hypothesis) {
                    current_state_->add_hypothesis(hp);
                  }
                  last_state_ = current_state_;
                  // Init tracking
                  belief_tracker_.belief_update(*last_state_, *current_state_);
                  }

    // Reward, Cost, Terminal, Collision, GoalReached, MaxSteps
    typedef std::tuple<float, float, bool, bool, bool> StepResult;
    StepResult step() {
      if(current_state_->is_terminal()) {
        return std::tuple<float, float, bool, bool, bool>();
      }
      std::vector<Reward> rewards;
      Cost cost;

      JointAction jointaction(current_state_->get_agent_idx().size());
      for (auto agent_idx : current_state_->get_agent_idx()) {
        if (agent_idx == CrossingState<Domain>::ego_agent_idx ) {
          // Plan for ego agent with hypothesis-based search
          Mcts<CrossingState<Domain>, UctStatistic, HypothesisStatistic, RandomHeuristic> mcts(mcts_parameters_);
          mcts.search(*current_state_, belief_tracker_, 5000, 10000);
          jointaction[agent_idx] = mcts.returnBestAction();
          std::cout << "best uct action: " << idx_to_ego_crossing_action<Domain>(jointaction[agent_idx]) << std::endl;
        } else {
          // Other agents act according to unknown true agents policy
          const auto action = agents_true_policies_.at(agent_idx).act(current_state_->get_agent_state(agent_idx),
                                                      current_state_->get_ego_state().x_pos);
          jointaction[agent_idx] = aconv(action);
        }
      }

      last_state_ = current_state_;
      current_state_ = last_state_->execute(jointaction, rewards, cost);
      belief_tracker_.belief_update(*last_state_, *current_state_);
      
      bool collision = current_state_->is_terminal() && !current_state_->ego_goal_reached();
      bool goal_reached = current_state_->ego_goal_reached();

      if(viewer_) {
        current_state_->draw(viewer_);
      }

      return std::make_tuple<float, float,bool, bool, bool> (rewards[CrossingState<Domain>::ego_agent_idx], 
                                                              cost,
                                                              std::move(current_state_->is_terminal()),
                                                              std::move(collision),
                                                              std::move(goal_reached));
    }

    static const std::vector<std::string> EVAL_RESULT_COLUMN_DESC;
    typedef std::tuple<float, float, bool, bool, bool, bool, unsigned int> EpisodeResult;
    EpisodeResult run() {
      unsigned int current_step=0;
      bool done = false;
      while(!done) {
        const auto step_result = step();
        const bool max_steps_reached = current_step > max_steps_;
        const bool terminal_state = std::get<2>(step_result);
        if(terminal_state || max_steps_reached) {
          return std::tuple_cat(step_result, std::forward_as_tuple(max_steps_reached),
                                 std::forward_as_tuple(current_step));
        }
        current_step += 1;
      }
    }

  private:
    Viewer* viewer_;
    std::shared_ptr<CrossingState<Domain>> current_state_;
    std::shared_ptr<CrossingState<Domain>> last_state_;
    HypothesisBeliefTracker belief_tracker_; // todo: pass params
    const std::unordered_map<AgentIdx, AgentPolicyCrossingState<Domain>> agents_true_policies_;
    const unsigned int max_steps_;
    const unsigned int mcts_max_search_time_;
    const unsigned int mcts_max_iterations_;
    const MctsParameters mcts_parameters_;
};


} // namespace mcts

#endif // MCTS_EPISODE_RUNNER_H_