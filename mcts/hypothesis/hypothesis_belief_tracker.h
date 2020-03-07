// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_HYPOTHESIS_HYPOTHESISBELIEFTRACKER_H
#define MCTS_HYPOTHESIS_HYPOTHESISBELIEFTRACKER_H

#include <random>
#include <deque>

#include "mcts/mcts_parameters.h"
#include "mcts/hypothesis/common.h"
#include "mcts/random_generator.h"
#include "mcts/hypothesis/hypothesis_state.h"


namespace mcts {


class HypothesisBeliefTracker : public mcts::RandomGenerator {
  public:
    typedef enum PosteriorType {
    PRODUCT = 0,
    SUM = 1
    } PosteriorType;


    HypothesisBeliefTracker(const MctsParameters& mcts_parameters) : 
                            RandomGenerator(mcts_parameters.hypothesis_belief_tracker.RANDOM_SEED_HYPOTHESIS_SAMPLING),
                            history_length_(mcts_parameters.hypothesis_belief_tracker.HISTORY_LENGTH),
                            probability_discount_(mcts_parameters.hypothesis_belief_tracker.PROBABILITY_DISCOUNT),
                            posterior_type_(static_cast<PosteriorType>(mcts_parameters.hypothesis_belief_tracker.POSTERIOR_TYPE)),
                            fixed_hypothesis_set_(static_cast<std::unordered_map<AgentIdx, HypothesisId>>(
                              mcts_parameters.hypothesis_belief_tracker.FIXED_HYPOTHESIS_SET)),
                            tracked_probabilities_(),
                            tracked_beliefs_(),
                            current_sampled_hypothesis_() {};

    template <typename S>
    void belief_update(const HypothesisStateInterface<S>& state,
                      const HypothesisStateInterface<S>& next_state);

    const std::unordered_map<AgentIdx, HypothesisId>& sample_current_hypothesis(); // shared across all states

    const std::unordered_map<AgentIdx, std::vector<Belief>> get_beliefs() const {
      return tracked_beliefs_;
    }

    std::string sprintf() const;

private:
    unsigned int history_length_;
    float probability_discount_;
    PosteriorType posterior_type_;
    std::unordered_map<AgentIdx, std::vector<std::deque<Probability>>> tracked_probabilities_;
    std::unordered_map<AgentIdx, std::vector<Belief>> tracked_beliefs_;//< contains the beliefs for each hypothesis for each agent 
    std::unordered_map<AgentIdx, HypothesisId> current_sampled_hypothesis_; //< the currently sampled hypothesis shared across all hypothesis states
    std::unordered_map<AgentIdx, HypothesisId> fixed_hypothesis_set_; // < if not empty a fixed hypothesis set is used in each iteration (e.g. for the omniscient approach)
};


template <typename S>
void HypothesisBeliefTracker::belief_update(const HypothesisStateInterface<S>& state, 
                                            const HypothesisStateInterface<S>& next_state) {
  if(!fixed_hypothesis_set_.empty()) {
    // no belief update required, if fixed hypothesis set is given
    return;
  }

  for(auto agent_idx : state.get_agent_idx() ) {
    auto belief_track_it = tracked_beliefs_.find(agent_idx);
    if(belief_track_it == tracked_beliefs_.end()) {
      // Init belief and probability tracking
      auto& belief_track_agent = tracked_beliefs_[agent_idx];
      auto& probability_track_agent = tracked_probabilities_[agent_idx];
      const auto num_hypothesis = state.get_num_hypothesis(agent_idx);
      for (HypothesisId hid = 0; hid < num_hypothesis; ++hid) {
        belief_track_agent.push_back(0.0f); // use as default but overwritten later
        probability_track_agent.push_back(std::deque<Probability>());
      }
    }

    // Update belief for each tracked hypothesis
    auto& belief_track_agent = tracked_beliefs_[agent_idx];
    auto& probability_track_agent = tracked_probabilities_[agent_idx];
    float belief_sum = 0.0f;
    for (HypothesisId hid = 0; hid < belief_track_agent.size(); ++hid) {
        // add latest hypothesis probability 
        probability_track_agent[hid].push_back(state.template get_probability_last_action<typename S::ActionType>(hid, agent_idx));
        if (probability_track_agent[hid].size()>history_length_) {
          probability_track_agent[hid].pop_front();
        }

        // calculate belief
        Probability belief;
        if(posterior_type_ == PosteriorType::PRODUCT) {
            belief = state.get_prior(hid, agent_idx);
          } else if(posterior_type_ == PosteriorType::SUM) {
            belief = 0.0001f; // some small value to initialize sum
          }
        float current_pdiscount = probability_discount_;
        const auto& probabilities = probability_track_agent[hid];
        //std::cout << "Update ------------------" << std::endl;
        for(auto pit = probabilities.rbegin(); pit != probabilities.rend(); ++pit) {
          //std::cout << "Probabilities: hyp = " << hid << "agent =" << int(agent_idx) << ", P=" << *pit<< std::endl;
          if(posterior_type_ == PosteriorType::PRODUCT) {
            belief *= *pit*current_pdiscount;
          } else if(posterior_type_ == PosteriorType::SUM) {
            belief += *pit*current_pdiscount;
          }
          current_pdiscount = current_pdiscount*probability_discount_;
        }
        belief_track_agent[hid] = belief;
        belief_sum += belief_track_agent[hid];
    }

    // Normalize beliefs
    if(belief_sum > 0.0f) {
      for (HypothesisId hid = 0; hid < belief_track_agent.size(); ++hid) {
          belief_track_agent[hid] /= belief_sum;
      }
  }
  }
}

inline const std::unordered_map<AgentIdx, HypothesisId>& HypothesisBeliefTracker::sample_current_hypothesis() {
  if(!fixed_hypothesis_set_.empty()) {
    current_sampled_hypothesis_ = fixed_hypothesis_set_;
    return current_sampled_hypothesis_;
  }

  for (const auto& it : tracked_beliefs_) {
    // Sample one hypothesis for each agent
    std::discrete_distribution<HypothesisId> hypothesis_distribution(it.second.begin(), it.second.end());
    auto& hypothesis_id = current_sampled_hypothesis_[it.first];
    hypothesis_id = hypothesis_distribution(random_generator_);
  }
  return current_sampled_hypothesis_;
}

inline std::string HypothesisBeliefTracker::sprintf() const {
  std::stringstream ss;
  // Beliefs
  ss << "Beliefs: " << std::endl;
  for (const auto& belief : get_beliefs()) {
    ss << "Ag. "  << belief.first << ": ";
    for (const auto& probability: belief.second) {
        ss << probability << ",";
    }
    ss << std::endl;
  }
  return ss.str();
}

} // namespace mcts

#endif // MCTS_HYPOTHESIS_HYPOTHESISBELIEFTRACKER_H