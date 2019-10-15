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


typedef int CrossingStateAction;

struct CrossingStateParameters {
    static int MAX_VELOCITY_OTHER;
    static int MIN_VELOCITY_OTHER;
    static int NUM_OTHER_ACTIONS() { return MAX_VELOCITY_OTHER-MIN_VELOCITY_OTHER + 1; }
    static int MAX_VELOCITY_EGO;
    static int MIN_VELOCITY_EGO;
    static int NUM_EGO_ACTIONS() { return MAX_VELOCITY_EGO - MIN_VELOCITY_EGO + 1; }
    static int CHAIN_LENGTH; 
    static int EGO_GOAL_POS;
    static int CROSSING_POINT() { return (CHAIN_LENGTH-1)/2+1; }
};

using CSP = CrossingStateParameters;

inline CrossingStateAction idx_to_ego_crossing_action(const ActionIdx& action) {
    // First action indices are for braking starting from zero
    return action + CSP::MIN_VELOCITY_EGO;
}


inline CrossingStateAction aconv(const ActionIdx& action) {
    return ((union { CrossingStateAction i; ActionIdx u; }){ .u = action }).i;
}

inline ActionIdx aconv(const CrossingStateAction& action) {
    return ((union { CrossingStateAction i; ActionIdx u; }){ .i = action }).u;
}

typedef struct AgentState {
    AgentState() : x_pos(0), last_action(0) {}
    AgentState(const int& x, const CrossingStateAction& last_action) : 
            x_pos(x), last_action(last_action) {}
    int x_pos;
    CrossingStateAction last_action;
} AgentState;

class AgentPolicyCrossingState : public RandomGenerator {
  public:
    AgentPolicyCrossingState(const std::pair<int, int>& desired_gap_range) : 
                            desired_gap_range_(desired_gap_range) {}
    CrossingStateAction act(const AgentState& agent_state, const int& ego_pos) const {
        // sample desired gap parameter
        std::uniform_int_distribution<int> dis(desired_gap_range_.first, desired_gap_range_.second);
        int desired_gap_dst = dis(random_generator_);

        return calculate_action(agent_state, ego_pos, desired_gap_dst);
    }

    CrossingStateAction calculate_action(const AgentState& agent_state, const int& ego_pos, const int& desired_gap_dst) const {
        // If past crossing point, use last execute action
        if(agent_state.x_pos < CSP::CROSSING_POINT() ) {        
            const auto gap_error = ego_pos - agent_state.x_pos - desired_gap_dst;
            // gap_error < 0 -> brake to increase distance
            if (desired_gap_dst > 0) {
                if(gap_error < 0) {
                    return std::max(gap_error, CSP::MIN_VELOCITY_OTHER);
                } else {
                    return std::min(gap_error, CSP::MAX_VELOCITY_OTHER);
                }
            } else {
                // Dont brake again if agents is already ahead of ego agent, but continue with same velocity
                return std::max(std::min(gap_error, CSP::MAX_VELOCITY_OTHER), agent_state.last_action);
            }
        } else {
            return agent_state.last_action;
        }

    }

    Probability get_probability(const AgentState& agent_state, const int& ego_pos, const CrossingStateAction& action) const {
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

  private: 
        const std::pair<int, int> desired_gap_range_;
};

// A simple environment with a 1D state, only if both agents select different actions, they get nearer to the terminal state
class CrossingState : public mcts::HypothesisStateInterface<CrossingState>
{
private:
  static const unsigned int num_other_agents = 2;

public:
    CrossingState(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis) :
                            HypothesisStateInterface<CrossingState>(current_agents_hypothesis),
                            hypothesis_(),
                            other_agent_states_(),
                            ego_state_(),
                            terminal_(false) {
                                for (auto& state : other_agent_states_) {
                                    state = AgentState();
                                }
                            }

    CrossingState(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis,
                            const std::array<AgentState, num_other_agents>& other_agent_states,
                            const AgentState& ego_state,
                            const bool& terminal,
                            const std::vector<AgentPolicyCrossingState>& hypothesis) : // add hypothesis to execute copying
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
        const HypothesisId agt_hyp_id = current_agents_hypothesis_.at(agent_idx);
        return aconv(hypothesis_.at(agt_hyp_id).act(other_agent_states_[agent_idx-1],
                                                    ego_state_.x_pos));
    };

    template<typename ActionType = CrossingStateAction>
    Probability get_probability(const HypothesisId& hypothesis, const AgentIdx& agent_idx, const CrossingStateAction& action) const { 
        return hypothesis_.at(hypothesis).get_probability(other_agent_states_[agent_idx-1], ego_state_.x_pos, action);
    ;}

    template<typename ActionType = CrossingStateAction>
    ActionType get_last_action(const AgentIdx& agent_idx) const {
        if (agent_idx == ego_agent_idx) {
            return ego_state_.last_action;
        } else {
            return other_agent_states_[agent_idx-1].last_action;
        }
    }

    Probability get_prior(const HypothesisId& hypothesis, const AgentIdx& agent_idx) const { return 0.5f;}

    HypothesisId get_num_hypothesis(const AgentIdx& agent_idx) const {return hypothesis_.size();}

    std::shared_ptr<CrossingState> execute(const JointAction& joint_action, std::vector<Reward>& rewards, Cost& ego_cost) const {
        // normally we map each single action value in joint action with a map to the floating point action. Here, not required
        
        const int old_x_ego = ego_state_.x_pos;
        int new_x_ego = ego_state_.x_pos + idx_to_ego_crossing_action(joint_action[ego_agent_idx]);
        bool ego_out_of_map = false;
        if(new_x_ego < 0) {
            ego_out_of_map = true;
        }
        const AgentState next_ego_state(new_x_ego, aconv(joint_action[ego_agent_idx]));

        std::array<AgentState, num_other_agents> next_other_agent_states;
        bool collision = false;
        for(size_t i = 0; i < other_agent_states_.size(); ++i) {
            const auto& old_state = other_agent_states_[i];
            int new_x = old_state.x_pos + static_cast<int>(aconv(joint_action[i+1]));
            next_other_agent_states[i] = AgentState( (new_x>= 0) ? new_x : 0, aconv(joint_action[i+1]));

            // if ego state history encloses crossing point and other state history encloses crossing point
            // a collision occurs
            if(next_ego_state.x_pos >= CSP::CROSSING_POINT() &&  old_x_ego<= CSP::CROSSING_POINT() &&
              next_other_agent_states[i].x_pos >= CSP::CROSSING_POINT() && 
              other_agent_states_[i].x_pos <= CSP::CROSSING_POINT() ) {
                collision = true;
            }
        }

        const bool goal_reached = next_ego_state.x_pos >= CSP::EGO_GOAL_POS;

        const bool terminal = goal_reached || collision || ego_out_of_map;
        rewards.resize(num_other_agents+1);
        rewards[0] = goal_reached * 100.0f - 1000.0f * collision - 1000.0f * ego_out_of_map;
        ego_cost = collision * 1.0f;

        return std::make_shared<CrossingState>(current_agents_hypothesis_, next_other_agent_states, next_ego_state, terminal, hypothesis_);
    }

    ActionIdx get_num_actions(AgentIdx agent_idx) const {
        if(agent_idx == ego_agent_idx) {
            return CSP::NUM_EGO_ACTIONS();
        } else {
            return CSP::NUM_OTHER_ACTIONS();
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

    void add_hypothesis(const AgentPolicyCrossingState& hypothesis) {
        hypothesis_.push_back(hypothesis);
    }

    void clear_hypothesis() {
        hypothesis_.clear();
    }

    bool ego_goal_reached() const {
        return ego_state_.x_pos >= CSP::CROSSING_POINT();
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

    inline AgentState get_agent_state(const AgentIdx& agent_idx) const {
        return other_agent_states_[agent_idx-1];
    }

    inline AgentState get_ego_state() const {
        return ego_state_;
    }

    inline const std::array<AgentState, num_other_agents>& get_agent_states() const {
        return other_agent_states_;
    }

    inline int distance_to_ego(const AgentIdx& other_agent_idx) const {
        return ego_state_.x_pos - other_agent_states_[other_agent_idx].x_pos;
    }

    typedef CrossingStateAction ActionType;

    std::vector<AgentPolicyCrossingState> hypothesis_;

    std::array<AgentState, num_other_agents> other_agent_states_;
    AgentState ego_state_;
    bool terminal_;
};



#endif //CROSSING_STATE_H
