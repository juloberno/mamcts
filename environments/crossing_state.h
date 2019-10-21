// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef CROSSING_STATE_H
#define CROSSING_STATE_H

#include <iostream>
#include <random>
#include <unordered_map>
#include "mcts/hypothesis/hypothesis_state.h"

using namespace mcts;

namespace mcts {
    class Viewer;
}


template <typename Domain>
struct CrossingStateParameters {
    static Domain MAX_VELOCITY_OTHER;
    static Domain MIN_VELOCITY_OTHER;
    static Domain NUM_OTHER_ACTIONS() { return MAX_VELOCITY_OTHER-MIN_VELOCITY_OTHER + 1; }
    static Domain MAX_VELOCITY_EGO;
    static Domain MIN_VELOCITY_EGO;
    static Domain NUM_EGO_ACTIONS() { return MAX_VELOCITY_EGO - MIN_VELOCITY_EGO + 1; }
    static Domain CHAIN_LENGTH; 
    static Domain EGO_GOAL_POS;
    static Domain CROSSING_POINT() { return (CHAIN_LENGTH-1)/2+1; }
};

template <typename Domain>
using CSP = CrossingStateParameters<Domain>;

template <typename Domain>
inline Domain idx_to_ego_crossing_action(const ActionIdx& action) {
    // First action indices are for braking starting from zero
    return action + CSP<Domain>::MIN_VELOCITY_EGO;
}

template <typename Domain>
inline Domain aconv(const ActionIdx& action) {
    return ((union { Domain i; ActionIdx u; }){ .u = action }).i;
}

template <typename Domain>
inline ActionIdx aconv(const Domain& action) {
    return ((union { Domain i; ActionIdx u; }){ .i = action }).u;
}

template <typename Domain>
struct AgentState {
    AgentState() : x_pos(0), last_action(0) {}
    AgentState(const Domain& x, const Domain& last_action) : 
            x_pos(x), last_action(last_action) {}
    Domain x_pos;
    Domain last_action;
};

template <typename Domain>
class AgentPolicyCrossingState : public RandomGenerator {
  public:
    AgentPolicyCrossingState(const std::pair<Domain, Domain>& desired_gap_range) : 
                            desired_gap_range_(desired_gap_range) {
                                MCTS_EXPECT_TRUE(desired_gap_range.first <= desired_gap_range.second)
                            }

    Domain act(const AgentState<Domain>& agent_state, const Domain& ego_pos) const;

    Probability get_probability(const AgentState<Domain>& agent_state, const Domain& ego_pos, const Domain& action) const;

    Domain calculate_action(const AgentState<Domain>& agent_state, const Domain& ego_pos, const Domain& desired_gap_dst) const {
        // If past crossing point, use last execute action
        if(agent_state.x_pos < CSP<Domain>::CROSSING_POINT() ) {        
            const auto gap_error = ego_pos - agent_state.x_pos - desired_gap_dst;
            // gap_error < 0 -> brake to increase distance
            if (desired_gap_dst > 0) {
                if(gap_error < 0) {
                    return std::max(gap_error, CSP<Domain>::MIN_VELOCITY_OTHER);
                } else {
                    return std::min(gap_error, CSP<Domain>::MAX_VELOCITY_OTHER);
                }
            } else {
                // Dont brake again if agents is already ahead of ego agent, but continue with same velocity
                return std::max(std::min(gap_error, CSP<Domain>::MAX_VELOCITY_OTHER), agent_state.last_action);
            }
        } else {
            return agent_state.last_action;
        }

    }

  private: 
        const std::pair<Domain, Domain> desired_gap_range_;
};

template <>
inline int AgentPolicyCrossingState<int>::act(const AgentState<int>& agent_state, const int& ego_pos) const {
    // sample desired gap parameter
    std::uniform_int_distribution<int> dis(desired_gap_range_.first, desired_gap_range_.second);
    int desired_gap_dst = dis(random_generator_);

    return calculate_action(agent_state, ego_pos, desired_gap_dst);
}

template <>
inline float AgentPolicyCrossingState<float>::act(const AgentState<float>& agent_state, const float& ego_pos) const {
    // sample desired gap parameter
    std::uniform_real_distribution<float> dis(desired_gap_range_.first, desired_gap_range_.second);
    int desired_gap_dst = dis(random_generator_);

    return calculate_action(agent_state, ego_pos, desired_gap_dst);
}


template <>
inline Probability AgentPolicyCrossingState<int>::get_probability(const AgentState<int>& agent_state, const int& ego_pos, const int& action) const {
    std::vector<int> gap_distances(desired_gap_range_.second - desired_gap_range_.first+1);
    std::iota(gap_distances.begin(), gap_distances.end(),desired_gap_range_.first);
    unsigned int action_selected = 0;
    for(const auto& desired_gap_dst : gap_distances) {
        const auto calculated = calculate_action(agent_state, ego_pos, desired_gap_dst);
        if(calculated == action ) {
            action_selected++;
        }
    }
    const auto probability = static_cast<float>(action_selected)/static_cast<float>(gap_distances.size());
    return probability;
}

template <>
inline Probability AgentPolicyCrossingState<float>::get_probability(const AgentState<float>& agent_state, const float& ego_pos, const float& action) const {
    const float uniform_prob = 1/(desired_gap_range_.second-desired_gap_range_.first);
    const float zero_prob = 0.0f;
    const float one_prob = 1.0f;

    if(agent_state.x_pos < CSP<float>::CROSSING_POINT() ) {        
        const auto gap_error_min = ego_pos - agent_state.x_pos - desired_gap_range_.first;
        const auto gap_error_max = ego_pos - agent_state.x_pos - desired_gap_range_.second;

        // gap_error < 0 -> brake to increase distance
        if ( desired_gap_range_.first >= 0 && desired_gap_range_.second > 0) {
            // For both boundaries of gap range gap error is negative 
            if(gap_error_min < 0 && gap_error_max <= 0 &&
                action <= std::max(gap_error_min, CSP<float>::MIN_VELOCITY_OTHER) &&
                 action >= std::max(gap_error_max, CSP<float>::MIN_VELOCITY_OTHER) ) {
                     // Consider boundaries due to maximum and minimum operations
                    if(gap_error_max < CSP<float>::MIN_VELOCITY_OTHER &&
                    gap_error_min < CSP<float>::MIN_VELOCITY_OTHER &&
                        action == CSP<float>::MIN_VELOCITY_OTHER) {
                        return one_prob;
                    } else if(gap_error_max < CSP<float>::MIN_VELOCITY_OTHER &&
                        action == CSP<float>::MIN_VELOCITY_OTHER) {
                            return (CSP<float>::MIN_VELOCITY_OTHER-gap_error_max)*uniform_prob;
                     } else if(gap_error_min < CSP<float>::MIN_VELOCITY_OTHER &&
                        action == CSP<float>::MIN_VELOCITY_OTHER) {
                            return (CSP<float>::MIN_VELOCITY_OTHER-gap_error_min)*uniform_prob;
                    } else {
                            return uniform_prob;
                    }
            // For only the higher desired gap the gap error is negative -> 
            } else if(gap_error_min >= 0 && gap_error_max <= 0 &&
                action <= std::min(gap_error_min, CSP<float>::MAX_VELOCITY_OTHER) &&
                action >= std::max(gap_error_max, CSP<float>::MIN_VELOCITY_OTHER) ) 
            {
                // Consider boundaries due to maximum and minimum operations
                if(gap_error_min > CSP<float>::MAX_VELOCITY_OTHER &&
                   action == CSP<float>::MAX_VELOCITY_OTHER) {
                   return (gap_error_min - CSP<float>::MAX_VELOCITY_OTHER)*uniform_prob;
                } else if(gap_error_max < CSP<float>::MIN_VELOCITY_OTHER &&
                          action == CSP<float>::MIN_VELOCITY_OTHER ) {
                    return (CSP<float>::MIN_VELOCITY_OTHER - gap_error_max)*uniform_prob;
                } else {
                    return uniform_prob;
                }
            // For both desired gap boundaries the gap error is positive (gap boundaries are ordered)
            } else if (gap_error_min < 0 && gap_error_max < 0 &&
                action <= std::min(gap_error_min, CSP<float>::MAX_VELOCITY_OTHER) &&
                action >= std::min(gap_error_max, CSP<float>::MAX_VELOCITY_OTHER) ) {
                // Consider boundaries due to maximum and minimum operations
                if(gap_error_min > CSP<float>::MAX_VELOCITY_OTHER &&
                        gap_error_max > CSP<float>::MAX_VELOCITY_OTHER &&
                        action == CSP<float>::MAX_VELOCITY_OTHER) {
                    return one_prob;
                } else if(gap_error_min > CSP<float>::MAX_VELOCITY_OTHER &&
                        action == CSP<float>::MAX_VELOCITY_OTHER) {
                    return (CSP<float>::MAX_VELOCITY_OTHER - gap_error_min)*uniform_prob;
                } else if(gap_error_max > CSP<float>::MAX_VELOCITY_OTHER &&
                        action == CSP<float>::MAX_VELOCITY_OTHER) {
                    return (gap_error_max - CSP<float>::MAX_VELOCITY_OTHER)*uniform_prob;
                } else {
                    return uniform_prob;

                }
            } else {
                return zero_prob;
            }
        } else if(desired_gap_range_.first < 0 && desired_gap_range_.second < 0 &&
            action >= std::max(std::min(gap_error_min, CSP<float>::MAX_VELOCITY_OTHER), agent_state.last_action) &&
            action <= std::max(std::min(gap_error_max, CSP<float>::MAX_VELOCITY_OTHER), agent_state.last_action) ) {
            // Dont brake again if agents is already ahead of ego agent, but continue with same velocity
            throw "not implemented. resolve max, min operation";
        } else {
            throw "probability calculation for mixed positive/negative gap range not implemented.";
        }
    } else {
        if(action == agent_state.last_action) {
            return one_prob;
        } else {
            return zero_prob;
        }
    }
}

// A simple environment with a 1D state, only if both agents select different actions, they get nearer to the terminal state
template <typename Domain>
class CrossingState : public mcts::HypothesisStateInterface<CrossingState<Domain>>
{
private:
  static const unsigned int num_other_agents = 2;

public:
    CrossingState(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis) :
                            HypothesisStateInterface<CrossingState<Domain>>(current_agents_hypothesis),
                            hypothesis_(),
                            other_agent_states_(),
                            ego_state_(),
                            terminal_(false) {
                                for (auto& state : other_agent_states_) {
                                    state = AgentState<Domain>();
                                }
                            }

    CrossingState(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis,
                            const std::array<AgentState<Domain>, num_other_agents>& other_agent_states,
                            const AgentState<Domain>& ego_state,
                            const bool& terminal,
                            const std::vector<AgentPolicyCrossingState<Domain>>& hypothesis) : // add hypothesis to execute copying
                            HypothesisStateInterface<CrossingState>(current_agents_hypothesis),
                            hypothesis_(hypothesis),
                            other_agent_states_(other_agent_states),
                            ego_state_(ego_state),
                            terminal_(terminal) {};
    ~CrossingState() {};

    std::shared_ptr<CrossingState> clone() const
    {
        return std::make_shared<CrossingState>(*this);
    }

    ActionIdx plan_action_current_hypothesis(const AgentIdx& agent_idx) const {
        const HypothesisId agt_hyp_id = this->current_agents_hypothesis_.at(agent_idx);
        return aconv(hypothesis_.at(agt_hyp_id).act(other_agent_states_[agent_idx-1],
                                                    ego_state_.x_pos));
    };

    template<typename ActionType = Domain>
    Probability get_probability(const HypothesisId& hypothesis, const AgentIdx& agent_idx, const Domain& action) const { 
        return hypothesis_.at(hypothesis).get_probability(other_agent_states_[agent_idx-1], ego_state_.x_pos, action);
    ;}

    template<typename ActionType = Domain>
    ActionType get_last_action(const AgentIdx& agent_idx) const {
        if (agent_idx == this->ego_agent_idx) {
            return ego_state_.last_action;
        } else {
            return other_agent_states_[agent_idx-1].last_action;
        }
    }

    Probability get_prior(const HypothesisId& hypothesis, const AgentIdx& agent_idx) const { return 0.5f;}

    HypothesisId get_num_hypothesis(const AgentIdx& agent_idx) const {return hypothesis_.size();}

    std::shared_ptr<CrossingState<Domain>> execute(const JointAction& joint_action, std::vector<Reward>& rewards, Cost& ego_cost) const {
        // normally we map each single action value in joint action with a map to the floating point action. Here, not required
        
        const auto old_x_ego = ego_state_.x_pos;
        auto new_x_ego = ego_state_.x_pos + idx_to_ego_crossing_action<Domain>(joint_action[this->ego_agent_idx]);
        bool ego_out_of_map = false;
        if(new_x_ego < 0) {
            ego_out_of_map = true;
        }
        const AgentState<Domain> next_ego_state(new_x_ego, aconv<Domain>(joint_action[this->ego_agent_idx]));

        std::array<AgentState<Domain>, num_other_agents> next_other_agent_states;
        bool collision = false;
        for(size_t i = 0; i < other_agent_states_.size(); ++i) {
            const auto& old_state = other_agent_states_[i];
            auto new_x = old_state.x_pos + static_cast<Domain>(aconv<Domain>(joint_action[i+1]));
            next_other_agent_states[i] = AgentState<Domain>( (new_x>= 0) ? new_x : 0, aconv<Domain>(joint_action[i+1]));

            // if ego state history encloses crossing point and other state history encloses crossing point
            // a collision occurs
            if(next_ego_state.x_pos >= CSP<Domain>::CROSSING_POINT() &&  old_x_ego<= CSP<Domain>::CROSSING_POINT() &&
              next_other_agent_states[i].x_pos >= CSP<Domain>::CROSSING_POINT() && 
              other_agent_states_[i].x_pos <= CSP<Domain>::CROSSING_POINT() ) {
                collision = true;
            }
        }

        const bool goal_reached = next_ego_state.x_pos >= CSP<Domain>::EGO_GOAL_POS;

        const bool terminal = goal_reached || collision || ego_out_of_map;
        rewards.resize(num_other_agents+1);
        rewards[0] = goal_reached * 100.0f - 1000.0f * collision - 1000.0f * ego_out_of_map;
        ego_cost = collision * 1.0f;

        return std::make_shared<CrossingState<Domain>>(this->current_agents_hypothesis_, next_other_agent_states, next_ego_state, terminal, hypothesis_);
    }

    ActionIdx get_num_actions(AgentIdx agent_idx) const {
        if(agent_idx == this->ego_agent_idx) {
            return CSP<Domain>::NUM_EGO_ACTIONS();
        } else {
            return CSP<Domain>::NUM_OTHER_ACTIONS();
        }
    }

    bool is_terminal() const {
        return terminal_;
    }

    const std::vector<AgentIdx> get_agent_idx() const {
        std::vector<AgentIdx> agent_idx(num_other_agents+1);
        std::iota(agent_idx.begin(), agent_idx.end(),0);
        return agent_idx; // adapt to number of agents
    }

    std::string sprintf() const
    {
        std::stringstream ss;
        ss << "Ego: x=" << ego_state_.x_pos;
        int i = 0;
        for (const auto& st : other_agent_states_) {
            ss << ", Ag" << i << ": x=" << st.x_pos;
            i++;
        } 
        ss << std::endl;
        return ss.str();
    }

    void add_hypothesis(const AgentPolicyCrossingState<Domain>& hypothesis) {
        hypothesis_.push_back(hypothesis);
    }

    void clear_hypothesis() {
        hypothesis_.clear();
    }

    bool ego_goal_reached() const {
        return ego_state_.x_pos >= CSP<Domain>::CROSSING_POINT();
    }

    int min_distance_to_ego() const {
        int min_dist = std::numeric_limits<int>::max();
        for (int i = 0; i < other_agent_states_.size(); ++i) {
            const auto dist = distance_to_ego(i);
            if (min_dist > dist) {
                min_dist = dist;
            }
        }
        return min_dist;
    }

    inline AgentState<Domain> get_agent_state(const AgentIdx& agent_idx) const {
        return other_agent_states_[agent_idx-1];
    }

    inline AgentState<Domain> get_ego_state() const {
        return ego_state_;
    }

    inline const std::array<AgentState<Domain>, num_other_agents>& get_agent_states() const {
        return other_agent_states_;
    }

    inline int distance_to_ego(const AgentIdx& other_agent_idx) const {
        return ego_state_.x_pos - other_agent_states_[other_agent_idx].x_pos;
    }

    void draw(Viewer* viewer) const {}

    typedef Domain ActionType;
private:
    std::vector<AgentPolicyCrossingState<Domain>> hypothesis_;

    std::array<AgentState<Domain>, num_other_agents> other_agent_states_;
    AgentState<Domain> ego_state_;
    bool terminal_;
};



#endif //CROSSING_STATE_H
