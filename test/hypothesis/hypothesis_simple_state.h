// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef SIMPLESTATE_H
#define SIMPLESTATE_H

#include <iostream>
#include <random>
#include "mcts/hypothesis/hypothesis_state.h"

using namespace mcts;


typedef enum Actions {
        WAIT = 0,
        FORWARD = 1,
        BACKWARD = -1,
        NUM = 3
    } Actions;

class AgentPolicyCrossingState {
    AgentPolicyCrossingState(const&  std::pair<unsigned int, unsigned int> desired_gap_range) : 
                            desired_gap_range_(desired_gap_range) {}
    ActionIdx act(const unsigned int& ego_distance) {
        // sample desired gap parameter
        std::mt19937 gen(1000); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<unsigned int> dis(desired_gap_range_.first, first.second);
        const unsigned int desired_gap_dst = dis(gen);

        return calculate_action(ego_distance, desired_gap_dst);
    }

    ActionIdx calculate_action(const unsigned int& ego_distance, const unsigned int desired_gap_dst) {
        if (ego_distance - desired_gap_dst > 0) {
            return Actions::FORWARD;
        } else if (ego_distance - desired_gap_dst == 0) {
            return Actions::WAIT;
        } else {
            return Actions::BACKWARD;
        }
    }

    Probability get_probability(const unsigned int& ego_distance, const ActionIdx& action) {
        std::vector<unsigned int> gap_distances(desired_gap_range.second-desired_gap_range.first+1);
        std::iota(gap_distances.begin(), gap_distances.end(),desired_gap_range.first);
        unsigned int action_selected = 0;
        for(desired_gap_dst : gap_distances) {
            const auto calculated = calculate_action(ego_distance, desired_gap_dst);
            if(calculated == action ) {
                action_selected++;
            }
        }
        return static_cast<float>(action_selected)/static_cast<float>(gap_distances.size());
    }

    private: 
        const std::pair<unsigned int, unsigned int> desired_gap_range_;
}

// A simple environment with a 1D state, only if both agents select different actions, they get nearer to the terminal state
class HypothesisCrossingState : public mcts::HypothesisStateInterface<HypothesisCrossingState>
{
public:
    HypothesisCrossingState(const std::array<AgentState, num_other_agents> other_agent_states,
                            const AgentState& ego_state,
                            const bool& terminal) :
                            HypothesisStateInterface<HypothesisSimpleState>(current_agents_hypothesis),
                            other_agent_states_(other_agent_states),
                            ego_state_(ego_state),
                            terminal_(terminal),
                            ego_last_action_(Actions::WAIT) {};
    ~HypothesisSimpleState() {};

    std::shared_ptr<HypothesisCrossingState> clone() const
    {
        return std::make_shared<HypothesisCrossingState>(*this);
    }

    ActionIdx plan_action_current_hypothesis(const AgentIdx& agent_idx) const {
        const HypothesisId agt_hyp_id = current_agents_hypothesis_[agent_idx];
        return hypothesis_map_[agt_hyp_id].act(dst_to_ego(agent_idx-1));
    };

    template<typename ActionType = int>
    Probability get_probability(const HypothesisId& hypothesis, const AgentIdx& agent_idx, const ActionType& action) const { 
        return hypothesis_map_[hypothesis].get_probability(dst_to_ego(agent_idx-1), action);
    ;}

    template<typename ActionType = int>
    ActionType get_last_action(const AgentIdx& agent_idx) const {
        if (agent_idx == S::ego_agent_idx) {
            return ego_state.last_action;
        } else {
            return other_agent_states[agent_idx_-1].last_action;
        }
    }

    Probability get_prior(const HypothesisId& hypothesis, const AgentIdx& agent_idx) const { return 0.5f;}

    HypothesisId get_num_hypothesis(const AgentIdx& agent_idx) const {return hypothesis_map_.size();}

    std::shared_ptr<HypothesisSimpleState> execute(const JointAction& joint_action, std::vector<Reward>& rewards, const Cost& ego_cost) const {
        // normally we map each single action value in joint action with a map to the floating point action. Here, not required
        const AgentState next_ego_state {ego_state.x_pos + join_action[S::ego_agent_idx], join_action[S::ego_agent_idx]};

        std::array<AgentState, num_other_agents> next_other_agent_states;
        for(size_t i = 0; i < other_agent_states_.size(); ++i) {
            const auto& old_state = other_agent_states_[i];
            next_other_agent_states[i] = {old_state.x_pos + join_action[i+1], join_action[i+1]};
        }

        const bool goal_reached = ego_state.x_pos >= ego_goal_reached_position;
        bool collision = false;
        for (const auto& state: next_other_agent_states) {
            if(next_ego_state.x_pos == state.x_pos) {
                collision = true;
            }
        }

        const bool terminal = goal_reached || collision;
        rewards.resize(num_other_agents+1);
        rewards[0] = goal_reached * 100.0f - 1000.0f * collision;
        cost = collision * 1.0f;
    }

    ActionIdx get_num_actions(AgentIdx agent_idx) const {
        return Actions::NUM; // WAIT, FORWARD, BACKWARD
    }

    bool is_terminal() const {
        return terminal_;
    }

    constexpr std::vector<AgentIdx> get_agent_idx() const {
        std::vector<AgentIdx> agent_idx;
        std::iota(agent_idx.begind(), agent_idx.end(),0);
        return agent_idx; // adapt to number of agents
    }

    std::string sprintf() const
    {
        std::stringstream ss; // todo
        return ss.str();
    }

    inline unsigned int dst_to_ego(const AgentIdx& other_agent_idx) const {
        return ego_state.x_pos - other_agent_states[other_agent_idx];
    }

private:
    constexpr unsigned int num_other_agents = 1;
    constexpr unsigned int state_x_length = 41; /* 21 is crossing point (41-1)/2+1 */
    constexpr unsigned int ego_goal_reached_position = 35;
    constexpr unsigned int crossing_point = (state_x_length-1)/2+1;

    const auto hypothesis_map_ = std::unordered_map<HypothesisId, AgentPolicyCrossingState> {
        {0, AgentPolicyCrossingState({-4, 1})}, // aggressive agent type (negative desired gap leads to always forward moving)
        {1, AgentPolicyCrossingState({1, 5})}, // medium type
        {2, AgentPolicyCrossingState({6, 10})} // passive type
    };
    
    typedef struct AgentState {
        AgentState() : x_pos(0), last_action(Actions::WAIT) {}
        unsigned int x_pos;
        Action last_action;
    } AgentState;
    std::array<AgentState, num_other_agents> other_agent_states_;
    unsigned int ego_state_;
    bool terminal_;
};



#endif 
