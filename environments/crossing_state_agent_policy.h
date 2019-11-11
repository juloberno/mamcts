// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================


#ifndef MCTS_CROSSING_STATE_AGENT_POLICY_H_
#define MCTS_CROSSING_STATE_AGENT_POLICY_H_

#include <iostream>
#include <random>
#include <unordered_map>
#include "mcts/random_generator.h"
#include "environments/crossing_state_common.h"


namespace mcts {

template <typename Domain>
class AgentPolicyCrossingState : public RandomGenerator {
  public:
    AgentPolicyCrossingState(const std::pair<Domain, Domain>& desired_gap_range,
                            const CrossingStateParameters<Domain>& parameters) : 
                            RandomGenerator(parameters.OTHER_AGENTS_POLICY_RANDOM_SEED),
                            desired_gap_range_(desired_gap_range),
                            parameters_(parameters) {
                                MCTS_EXPECT_TRUE(desired_gap_range.first <= desired_gap_range.second)
                            }

    Domain act(const AgentState<Domain>& agent_state, const AgentState<Domain>& ego_state) const;

    Probability get_probability(const AgentState<Domain>& agent_state, const AgentState<Domain>& ego_state, const Domain& action) const;

    Domain calculate_action(const AgentState<Domain>& agent_state, const AgentState<Domain> ego_state, const Domain& desired_gap_dst) const {
        // If past crossing point, use last execute action
        if(agent_state.x_pos < parameters_.CROSSING_POINT() ) {
            // use a forward predicted ego state based on the last action
            const auto gap_error = ego_state.x_pos + ego_state.last_action - agent_state.x_pos - desired_gap_dst;
            // gap_error < 0 -> brake to increase distance
            if (desired_gap_dst > 0) {
                if(gap_error < 0) {
                    return std::max(gap_error, parameters_.MIN_VELOCITY_OTHER);
                } else {
                    return std::min(gap_error, parameters_.MAX_VELOCITY_OTHER);
                }
            } else {
                // Dont brake again if agents is already ahead of ego agent, but continue with same velocity
                return std::max(std::min(gap_error, parameters_.MAX_VELOCITY_OTHER), agent_state.last_action);
            }
        } else {
            return agent_state.last_action;
        }

    }

    std::string info() const {
      std::stringstream ss;
      ss << "[" << desired_gap_range_.first << ", " << desired_gap_range_.second << "]";
      return ss.str();
    }

  private: 
        const std::pair<Domain, Domain> desired_gap_range_;
        const CrossingStateParameters<Domain>& parameters_;
};

template <>
inline int AgentPolicyCrossingState<int>::act(const AgentState<int>& agent_state, const AgentState<int>& ego_state) const {
    // sample desired gap parameter
    std::uniform_int_distribution<int> dis(desired_gap_range_.first, desired_gap_range_.second);
    int desired_gap_dst = dis(this->random_generator_);

    return calculate_action(agent_state, ego_state, desired_gap_dst);
}

template <>
inline float AgentPolicyCrossingState<float>::act(const AgentState<float>& agent_state, const AgentState<float>& ego_state) const {
    // sample desired gap parameter
    std::uniform_real_distribution<float> dis(desired_gap_range_.first, desired_gap_range_.second);
    float desired_gap_dst = dis(this->random_generator_);

    return calculate_action(agent_state, ego_state, desired_gap_dst);
}


template <>
inline Probability AgentPolicyCrossingState<int>::get_probability(const AgentState<int>& agent_state, const AgentState<int>& ego_state, const int& action) const {
    std::vector<int> gap_distances(desired_gap_range_.second - desired_gap_range_.first+1);
    std::iota(gap_distances.begin(), gap_distances.end(),desired_gap_range_.first);
    unsigned int action_selected = 0;
    for(const auto& desired_gap_dst : gap_distances) {
        const auto calculated = calculate_action(agent_state, ego_state, desired_gap_dst);
        if(calculated == action ) {
            action_selected++;
        }
    }
    const auto probability = static_cast<float>(action_selected)/static_cast<float>(gap_distances.size());
    return probability;
}

template <>
inline Probability AgentPolicyCrossingState<float>::get_probability(const AgentState<float>& agent_state, const AgentState<float>& ego_state, const float& action) const {
    const float gap_discretization = 0.001f;
    MCTS_EXPECT_TRUE((desired_gap_range_.second-desired_gap_range_.first) > gap_discretization);
    MCTS_EXPECT_TRUE((desired_gap_range_.second-desired_gap_range_.first) % gap_discretization == 0);
    auto prob_between = [&](float left, float right, float uniform_prob) -> Probability {
        return std::max(right - left, gap_discretization) * uniform_prob;
    };
    const Probability zero_prob = 0.0f;
    const Probability one_prob = 1.0f;
   
    // Lambda function to calculate probability for positive gap errors    
    auto probability_positive_gap_error = [&](const float gap_error_min, const float gap_error_max,
                                              const float uniform_prob, const float single_sample_prob)
                                              -> Probability {
        // For both boundaries of gap range gap error is negative 
        if(gap_error_min < 0 && gap_error_max <= 0 &&
            action <= std::max(gap_error_min, parameters_.MIN_VELOCITY_OTHER) &&
                action >= std::max(gap_error_max, parameters_.MIN_VELOCITY_OTHER) ) {
                    // Consider boundaries due to maximum and minimum operations
                if(gap_error_max < parameters_.MIN_VELOCITY_OTHER &&
                gap_error_min < parameters_.MIN_VELOCITY_OTHER &&
                    action == parameters_.MIN_VELOCITY_OTHER) {
                    return one_prob;
                } else if(gap_error_max < parameters_.MIN_VELOCITY_OTHER &&
                    action == parameters_.MIN_VELOCITY_OTHER) {
                        return prob_between(gap_error_max, parameters_.MIN_VELOCITY_OTHER, uniform_prob);
                    } else if(gap_error_min < parameters_.MIN_VELOCITY_OTHER &&
                    action == parameters_.MIN_VELOCITY_OTHER) {
                        return prob_between(gap_error_min, parameters_.MIN_VELOCITY_OTHER, uniform_prob);
                } else {
                        return single_sample_prob;
                }
        // For only the higher desired gap the gap error is negative -> 
        } else if(gap_error_min >= 0 && gap_error_max <= 0 &&
            action <= std::min(gap_error_min, parameters_.MAX_VELOCITY_OTHER) &&
            action >= std::max(gap_error_max, parameters_.MIN_VELOCITY_OTHER) ) 
        {
            // Consider boundaries due to maximum and minimum operations
            if(gap_error_min > parameters_.MAX_VELOCITY_OTHER &&
                action == parameters_.MAX_VELOCITY_OTHER) {
                return prob_between(parameters_.MAX_VELOCITY_OTHER, gap_error_min, uniform_prob);
            } else if(gap_error_max < parameters_.MIN_VELOCITY_OTHER &&
                        action == parameters_.MIN_VELOCITY_OTHER ) {
                return prob_between(gap_error_max, parameters_.MIN_VELOCITY_OTHER, uniform_prob);
            } else {
                return single_sample_prob;
            }
        // For both desired gap boundaries the gap error is positive (gap boundaries are ordered)
        } else if (gap_error_min < 0 && gap_error_max < 0 &&
            action <= std::min(gap_error_min, parameters_.MAX_VELOCITY_OTHER) &&
            action >= std::min(gap_error_max, parameters_.MAX_VELOCITY_OTHER) ) {
            // Consider boundaries due to maximum and minimum operations
            if(gap_error_min > parameters_.MAX_VELOCITY_OTHER &&
                    gap_error_max > parameters_.MAX_VELOCITY_OTHER &&
                    action == parameters_.MAX_VELOCITY_OTHER) {
                return one_prob;
            } else if(gap_error_min > parameters_.MAX_VELOCITY_OTHER &&
                    action == parameters_.MAX_VELOCITY_OTHER) {
                return prob_between(parameters_.MAX_VELOCITY_OTHER, gap_error_min, uniform_prob);
            } else if(gap_error_max > parameters_.MAX_VELOCITY_OTHER &&
                    action == parameters_.MAX_VELOCITY_OTHER) {
                return prob_between(parameters_.MAX_VELOCITY_OTHER, gap_error_max, uniform_prob);
            } else {
                return single_sample_prob;
            }
        } else {
            return zero_prob;
        }
    };
    
    // Lambda function to calculate probabilities for negative gap errors
    auto probability_negative_gap_error = [&](const float gap_error_min, const float gap_error_max,
                                              const float uniform_prob, const float single_sample_prob)
                                              -> Probability {
        if(action >= std::max(std::min(gap_error_max, parameters_.MAX_VELOCITY_OTHER), agent_state.last_action) &&
        action <= std::max(std::min(gap_error_min, parameters_.MAX_VELOCITY_OTHER), agent_state.last_action) ) {
            // first check if action can only come up by using last action
            if (agent_state.last_action == action &&
                    std::min(gap_error_min, parameters_.MAX_VELOCITY_OTHER) <= agent_state.last_action &&
                    std::min(gap_error_max, parameters_.MAX_VELOCITY_OTHER) <= agent_state.last_action) {
                return one_prob;
            }
            // then resolve inner max operations, first if we took the last action ...
            else if(gap_error_min > parameters_.MAX_VELOCITY_OTHER &&
                gap_error_max > parameters_.MAX_VELOCITY_OTHER &&
                action == parameters_.MAX_VELOCITY_OTHER) {
                return one_prob;
            } else if(gap_error_min > parameters_.MAX_VELOCITY_OTHER &&
                        gap_error_max < parameters_.MAX_VELOCITY_OTHER &&
                    action == parameters_.MAX_VELOCITY_OTHER) {
                return prob_between(parameters_.MAX_VELOCITY_OTHER, gap_error_min, uniform_prob);
            } else if(gap_error_max > parameters_.MAX_VELOCITY_OTHER &&
                        gap_error_min < parameters_.MAX_VELOCITY_OTHER &&
                    action == parameters_.MAX_VELOCITY_OTHER) {
                return prob_between(parameters_.MAX_VELOCITY_OTHER, gap_error_max, uniform_prob);
            } else {
                return single_sample_prob;  
            }
        } else {
                return zero_prob;
        }
    }; 
        
    // Distinguish between the different cases 
    if(agent_state.x_pos < parameters_.CROSSING_POINT() ) {
        const auto gap_error_min = ego_state.x_pos + ego_state.last_action - agent_state.x_pos - desired_gap_range_.first;
        const auto gap_error_max = ego_state.x_pos + ego_state.last_action - agent_state.x_pos - desired_gap_range_.second;
        const auto gap_error_desired_gap_zero = ego_state.x_pos + ego_state.last_action - agent_state.x_pos;
        if ( desired_gap_range_.first >= 0 && desired_gap_range_.second > 0) {
            const Probability uniform_prob = 1/std::abs(desired_gap_range_.second-desired_gap_range_.first);
            const Probability single_sample_prob = uniform_prob * gap_discretization;
            return probability_positive_gap_error(gap_error_min, gap_error_max, uniform_prob, single_sample_prob);
        } else if(desired_gap_range_.first < 0 && desired_gap_range_.second <= 0) {
            const Probability uniform_prob = 1/std::abs(desired_gap_range_.second-desired_gap_range_.first);
            const Probability single_sample_prob = uniform_prob * gap_discretization;
            return probability_negative_gap_error(gap_error_min, gap_error_max, uniform_prob, single_sample_prob);
        } else if(desired_gap_range_.first < 0 && desired_gap_range_.second > 0) {
            // Split up gap range into negative and positive part and combine with probability OR
            const Probability uniform_prob_pos_range = 1/desired_gap_range_.second;
            const Probability uniform_prob_neg_range = 1/-desired_gap_range_.first;
            const Probability single_sample_prob_pos_range = uniform_prob_pos_range * gap_discretization;
            const Probability single_sample_prob_neg_range = uniform_prob_neg_range * gap_discretization;
            const Probability negative_range_prob = (-desired_gap_range_.first)/std::abs(desired_gap_range_.second-desired_gap_range_.first);
            const Probability positive_range_prob = (desired_gap_range_.second)/std::abs(desired_gap_range_.second-desired_gap_range_.first);
            
            const Probability p_negative_gap = probability_negative_gap_error(
                gap_error_min, gap_error_desired_gap_zero, uniform_prob_neg_range, single_sample_prob_neg_range
            );
            const Probability p_positive_gap = probability_positive_gap_error(
                gap_error_desired_gap_zero, gap_error_max, uniform_prob_pos_range, single_sample_prob_pos_range
            );
            return p_negative_gap*negative_range_prob + p_positive_gap*positive_range_prob;
        } else {
            throw "invalid configuration of desired gap range.";
        }
    } else {
        if(action == agent_state.last_action) {
            return one_prob;
        } else {
            return zero_prob;
        }
    }
}

} // namespace mcts

#endif // MCTS_CROSSING_STATE_AGENT_POLICY_H_