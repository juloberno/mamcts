// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_HYPOTHESIS_HYPOTHESISBELIEFTRACKER_H
#define MCTS_HYPOTHESIS_HYPOTHESISBELIEFTRACKER_H

#include <random>

#include "mcts/hypothesis/common.h"
#include "mcts/random_generator.h"
#include "mcts/hypothesis/hypothesis_state.h"


namespace mcts {

template <typename S>
class HypothesisBeliefTracker : public mcts::RandomGenerator {
  public:
    HypothesisBeliefTracker() : tracked_beliefs_(), current_sampled_hypothesis_() {};
    
    void belief_update(const HypothesisStateInterface<S>& state);

    const std::unordered_map<AgentIdx, HypothesisId>& sample_current_hypothesis(); // shared across all states

    const std::unordered_map<AgentIdx, std::vector<Belief>> get_beliefs() const {
      return tracked_beliefs_;
    }
private:
    std::unordered_map<AgentIdx, std::vector<Belief>> tracked_beliefs_;//< contains the beliefs for each hypothesis for each agent 
    std::unordered_map<AgentIdx, HypothesisId> current_sampled_hypothesis_;
};

template <typename S>
void HypothesisBeliefTracker<S>::belief_update(const HypothesisStateInterface<S>& state) {
  for(auto agent_idx : state.get_agent_idx() ) {
    auto belief_track_it = tracked_beliefs_.find(agent_idx);
    if(belief_track_it == tracked_beliefs_.end()) {
      // Init belief tracking
      auto& belief_track_agent = tracked_beliefs_[agent_idx];
      const auto num_hypothesis = state.get_num_hypothesis(agent_idx);
      for (HypothesisId hid = 0; hid < num_hypothesis; ++hid) {
        belief_track_agent.push_back(state.get_prior(hid, agent_idx));
      }
    }

    // Update belief for each tracked hypothesis
    auto& belief_track_agent = tracked_beliefs_[agent_idx];
    float belief_sum = 0.0f;
    for (HypothesisId hid = 0; hid < belief_track_agent.size(); ++hid) {
        const auto& last_action = state.get_last_action(agent_idx);
        belief_track_agent[hid] *= state.get_probability(hid, agent_idx, last_action);
        belief_sum += belief_track_agent[hid];
    }

    // Normalize beliefs
    for (HypothesisId hid = 0; hid < belief_track_agent.size(); ++hid) {
        belief_track_agent[hid] /= belief_sum;
    }
  }
}

template <typename S>
const std::unordered_map<AgentIdx, HypothesisId>& HypothesisBeliefTracker<S>::sample_current_hypothesis() {
  for (const auto& it : tracked_beliefs_) {
    // Sample one hypothesis for each agent
    std::discrete_distribution<HypothesisId> hypothesis_distribution(it.second.begin(), it.second.end());
    auto& hypothesis_id = current_sampled_hypothesis_[it.first];
    hypothesis_id = hypothesis_distribution(random_generator_);
  }
  return current_sampled_hypothesis_;
}

} // namespace mcts

#endif // MCTS_HYPOTHESIS_HYPOTHESISBELIEFTRACKER_H