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

#include "environments/viewer.h"
#include "environments/crossing_state_common.h"
#include "environments/crossing_state_parameters.h"
#include "environments/crossing_state_agent_policy.h"

namespace mcts {

// A simple environment with a 1D state, only if both agents select different actions, they get nearer to the terminal state
template <typename Domain>
class CrossingState : public mcts::HypothesisStateInterface<CrossingState<Domain>>
{
public:
    CrossingState(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis,
                  const CrossingStateParameters<Domain>& parameters) :
                            HypothesisStateInterface<CrossingState<Domain>>(current_agents_hypothesis),
                            hypothesis_(),
                            other_agent_states_(parameters.NUM_OTHER_AGENTS),
                            ego_state_(),
                            terminal_(false),
                            parameters_(parameters) {
                                for (auto& state : other_agent_states_) {
                                    state = AgentState<Domain>();
                                }
                            }

    CrossingState(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis,
                  const CrossingStateParameters<Domain>& parameters,
                  const std::vector<AgentState<Domain>>& other_agent_states,
                  const AgentState<Domain>& ego_state,
                  const bool& terminal,
                  const std::vector<AgentPolicyCrossingState<Domain>>& hypothesis
                  ) : 
                            HypothesisStateInterface<CrossingState>(current_agents_hypothesis),
                            hypothesis_(hypothesis),
                            other_agent_states_(other_agent_states),
                            ego_state_(ego_state),
                            terminal_(terminal),
                            parameters_(parameters) {};
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
        auto new_x_ego = ego_state_.x_pos + idx_to_ego_crossing_action(joint_action[this->ego_agent_idx]);
        bool ego_out_of_map = false;
        if(new_x_ego < 0) {
            ego_out_of_map = true;
        }
        const AgentState<Domain> next_ego_state(new_x_ego, aconv<Domain>(joint_action[this->ego_agent_idx]));

        std::vector<AgentState<Domain>> next_other_agent_states(other_agent_states_.size());
        bool collision = false;
        for(size_t i = 0; i < other_agent_states_.size(); ++i) {
            const auto& old_state = other_agent_states_[i];
            auto new_x = old_state.x_pos + static_cast<Domain>(aconv<Domain>(joint_action[i+1]));
            next_other_agent_states[i] = AgentState<Domain>( (new_x>= 0) ? new_x : 0, aconv<Domain>(joint_action[i+1]));

            // if ego state history encloses crossing point and other state history encloses crossing point
            // a collision occurs
            if(next_ego_state.x_pos >= parameters_.CROSSING_POINT() &&  old_x_ego<= parameters_.CROSSING_POINT() &&
              next_other_agent_states[i].x_pos >= parameters_.CROSSING_POINT() && 
              other_agent_states_[i].x_pos <= parameters_.CROSSING_POINT() ) {
                collision = true;
            }
        }

        const bool goal_reached = next_ego_state.x_pos >= parameters_.EGO_GOAL_POS;

        const bool terminal = goal_reached || collision || ego_out_of_map;
        rewards.resize(other_agent_states_.size()+1);
        rewards[0] = goal_reached * 100.0f - 1000.0f * collision - 1000.0f * ego_out_of_map;
        ego_cost = collision * 1.0f;

        return std::make_shared<CrossingState<Domain>>(this->current_agents_hypothesis_,
                                                       parameters_,
                                                       next_other_agent_states,
                                                       next_ego_state,
                                                       terminal,
                                                       hypothesis_);
    }

    ActionIdx get_num_actions(AgentIdx agent_idx) const {
        if(agent_idx == this->ego_agent_idx) {
            return parameters_.NUM_EGO_ACTIONS();
        } else {
            return parameters_.NUM_OTHER_ACTIONS();
        }
    }

    bool is_terminal() const {
        return terminal_;
    }

    const std::vector<AgentIdx> get_agent_idx() const {
        std::vector<AgentIdx> agent_idx(other_agent_states_.size()+1);
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
        return ego_state_.x_pos >= parameters_.CROSSING_POINT();
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

    inline const std::vector<AgentState<Domain>>& get_agent_states() const {
        return other_agent_states_;
    }

    inline int distance_to_ego(const AgentIdx& other_agent_idx) const {
        return ego_state_.x_pos - other_agent_states_[other_agent_idx].x_pos;
    }

    
    void draw(mcts::Viewer* viewer) const {
        // draw map ( crossing point is always at zero)
        const float state_draw_dst = 1.0f;
        const float linewidth = 2;
        const float state_draw_size = 50;
        const float factor_draw_current_state = 4;

        // draw lines equally spaced angles with small points
        // indicating states and larger points indicating the current state
        const float angle_delta = M_PI/(other_agent_states_.size()+2); // one for ego 
        const float line_radius = state_draw_dst*(parameters_.CHAIN_LENGTH-1)/2.0f;
        for(int i = 0; i < other_agent_states_.size()+1; ++i) {
            float start_angle = 1.5*M_PI - (i+1)*angle_delta;
            float end_angle = start_angle + M_PI;
            std::pair<float, float> line_x{cos(start_angle)*line_radius, cos(end_angle)*line_radius };
            std::pair<float, float> line_y{sin(start_angle)*line_radius, sin(end_angle)*line_radius};
            std::tuple<float,float,float,float> color{0,0,0,0};

            // Differentiate between ego and other agents
            AgentState<Domain> state;
            if(i == std::floor(other_agent_states_.size()/2)) {
                state = ego_state_;
                color = {0.8,0,0,0}; 
            } else {
                AgentIdx agt_idx = i;
                if (i > std::floor(other_agent_states_.size()/2)) {
                    agt_idx  = i-1;
                }
                state = other_agent_states_[agt_idx];
            }
            viewer->drawLine(line_x, line_y,
                linewidth, color);

            // Draw current states
            if(std::is_same<Domain, int>::value) {
                for (int y = 0; y < parameters_.CHAIN_LENGTH; ++y) {
                    const auto px = line_x.first + (line_x.second - line_x.first) * static_cast<float>(y) /
                                                                    static_cast<float>(parameters_.CHAIN_LENGTH-1);
                    const auto py = line_y.first + (line_y.second - line_y.first) * static_cast<float>(y) /
                                                                    static_cast<float>(parameters_.CHAIN_LENGTH-1);
                    float pointsize_temp = state_draw_size; 
                    if (state.x_pos == y) {
                        pointsize_temp *= factor_draw_current_state;
                    }
                    viewer->drawPoint(px, py,
                                pointsize_temp, color);
                }
            } else if (std::is_same<Domain, float>::value) {
                const auto px = line_x.first + (line_x.second - line_x.first) * state.x_pos /
                                                                    static_cast<float>(parameters_.CHAIN_LENGTH-1);
                const auto py = line_y.first + (line_y.second - line_y.first) * state.x_pos /
                                                                static_cast<float>(parameters_.CHAIN_LENGTH-1);
                float pointsize_temp = state_draw_size*factor_draw_current_state; 
                viewer->drawPoint(px, py,
                            pointsize_temp, color);
            } else {
                std::cout << "Unable to draw state information for a non-float and non-int state type." << std::endl;
            }
        }

    }

    Domain idx_to_ego_crossing_action(const ActionIdx& action) const {
        // First action indices are for braking starting from zero
        return action + parameters_.MIN_VELOCITY_EGO;
    }

    typedef Domain ActionType;
private:
    std::vector<AgentPolicyCrossingState<Domain>> hypothesis_;

    std::vector<AgentState<Domain>> other_agent_states_;
    AgentState<Domain> ego_state_;
    bool terminal_;

    const CrossingStateParameters<Domain>& parameters_;
};

} // namespace mcts

#endif //CROSSING_STATE_H
