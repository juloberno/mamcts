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
#include "mcts/random_generator.h"
#include "mcts/hypothesis/hypothesis_state.h"
#include "environments/viewer.h"
#include "environments/crossing_state_parameters.h"

using namespace mcts;


template <typename Domain>
using CSP = CrossingStateParameters<Domain>;

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
    AgentPolicyCrossingState(const std::pair<Domain, Domain>& desired_gap_range,
                            const CrossingStateParameters<Domain>& parameters) : 
                            desired_gap_range_(desired_gap_range),
                            parameters_(parameters) {
                                MCTS_EXPECT_TRUE(desired_gap_range.first <= desired_gap_range.second)
                            }

    Domain act(const AgentState<Domain>& agent_state, const Domain& ego_pos) const;

    Probability get_probability(const AgentState<Domain>& agent_state, const Domain& ego_pos, const Domain& action) const;

    Domain calculate_action(const AgentState<Domain>& agent_state, const Domain& ego_pos, const Domain& desired_gap_dst) const {
        // If past crossing point, use last execute action
        if(agent_state.x_pos < parameters_.CROSSING_POINT() ) {        
            const auto gap_error = ego_pos - agent_state.x_pos - desired_gap_dst;
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

  private: 
        const std::pair<Domain, Domain> desired_gap_range_;
        const CrossingStateParameters<Domain>& parameters_;
};

template <>
inline int AgentPolicyCrossingState<int>::act(const AgentState<int>& agent_state, const int& ego_pos) const {
    // sample desired gap parameter
    std::uniform_int_distribution<int> dis(desired_gap_range_.first, desired_gap_range_.second);
    int desired_gap_dst = dis(RandomGenerator::random_generator_);

    return calculate_action(agent_state, ego_pos, desired_gap_dst);
}

template <>
inline float AgentPolicyCrossingState<float>::act(const AgentState<float>& agent_state, const float& ego_pos) const {
    // sample desired gap parameter
    std::uniform_real_distribution<float> dis(desired_gap_range_.first, desired_gap_range_.second);
    int desired_gap_dst = dis(RandomGenerator::random_generator_);

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
    const float gap_discretization = 0.001f;
    MCTS_EXPECT_TRUE((desired_gap_range_.second-desired_gap_range_.first) > gap_discretization);
    MCTS_EXPECT_TRUE((desired_gap_range_.second-desired_gap_range_.first) % gap_discretization == 0);
    const float uniform_prob = 1/std::abs(desired_gap_range_.second-desired_gap_range_.first);
    const float single_sample_prob = uniform_prob * gap_discretization;
    auto prob_between = [&](float left, float right) {
        return std::max(right - left, gap_discretization) * uniform_prob;
    };
    const float zero_prob = 0.0f;
    const float one_prob = 1.0f;

    if(agent_state.x_pos < parameters_.CROSSING_POINT() ) {        
        const auto gap_error_min = ego_pos - agent_state.x_pos - desired_gap_range_.first;
        const auto gap_error_max = ego_pos - agent_state.x_pos - desired_gap_range_.second;

        // gap_error < 0 -> brake to increase distance
        if ( desired_gap_range_.first >= 0 && desired_gap_range_.second > 0) {
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
                            return prob_between(gap_error_max, parameters_.MIN_VELOCITY_OTHER);
                     } else if(gap_error_min < parameters_.MIN_VELOCITY_OTHER &&
                        action == parameters_.MIN_VELOCITY_OTHER) {
                            return prob_between(gap_error_min, parameters_.MIN_VELOCITY_OTHER);
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
                   return prob_between(parameters_.MAX_VELOCITY_OTHER, gap_error_min);
                } else if(gap_error_max < parameters_.MIN_VELOCITY_OTHER &&
                          action == parameters_.MIN_VELOCITY_OTHER ) {
                    return prob_between(gap_error_max, parameters_.MIN_VELOCITY_OTHER);
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
                    return prob_between(parameters_.MAX_VELOCITY_OTHER, gap_error_min);
                } else if(gap_error_max > parameters_.MAX_VELOCITY_OTHER &&
                        action == parameters_.MAX_VELOCITY_OTHER) {
                    return prob_between(parameters_.MAX_VELOCITY_OTHER, gap_error_max);
                } else {
                    return single_sample_prob;
                }
            } else {
                return zero_prob;
            }
        // hypothesis with both negative gap boundaries
        } else if(desired_gap_range_.first < 0 && desired_gap_range_.second < 0) {
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
                    return prob_between(parameters_.MAX_VELOCITY_OTHER, gap_error_min);
                } else if(gap_error_max > parameters_.MAX_VELOCITY_OTHER &&
                            gap_error_min < parameters_.MAX_VELOCITY_OTHER &&
                        action == parameters_.MAX_VELOCITY_OTHER) {
                    return prob_between(parameters_.MAX_VELOCITY_OTHER, gap_error_max);
                } else {
                    return single_sample_prob;  
                }
            } else {
                    return zero_prob;
            }
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
    CrossingState(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis,
                  const CrossingStateParameters<Domain>& parameters) :
                            HypothesisStateInterface<CrossingState<Domain>>(current_agents_hypothesis),
                            hypothesis_(),
                            other_agent_states_(),
                            ego_state_(),
                            terminal_(false),
                            parameters_(parameters) {
                                for (auto& state : other_agent_states_) {
                                    state = AgentState<Domain>();
                                }
                            }

    CrossingState(const std::unordered_map<AgentIdx, HypothesisId>& current_agents_hypothesis,
                  const CrossingStateParameters<Domain>& parameters,
                  const std::array<AgentState<Domain>, num_other_agents>& other_agent_states,
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

        std::array<AgentState<Domain>, num_other_agents> next_other_agent_states;
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
        rewards.resize(num_other_agents+1);
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

    inline const std::array<AgentState<Domain>, num_other_agents>& get_agent_states() const {
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
        const float angle_delta = M_PI/(num_other_agents+2); // one for ego 
        const float line_radius = state_draw_dst*(parameters_.CHAIN_LENGTH-1)/2.0f;
        for(int i = 0; i < num_other_agents+1; ++i) {
            float start_angle = 1.5*M_PI - (i+1)*angle_delta;
            float end_angle = start_angle + M_PI;
            std::pair<float, float> line_x{cos(start_angle)*line_radius, cos(end_angle)*line_radius };
            std::pair<float, float> line_y{sin(start_angle)*line_radius, sin(end_angle)*line_radius};
            std::tuple<float,float,float,float> color{0,0,0,0};

            // Differentiate between ego and other agents
            AgentState<Domain> state;
            if(i == std::floor(num_other_agents/2)) {
                state = ego_state_;
                color = {0.8,0,0,0}; 
            } else {
                AgentIdx agt_idx = i;
                if (i > std::floor(num_other_agents/2)) {
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

    std::array<AgentState<Domain>, num_other_agents> other_agent_states_;
    AgentState<Domain> ego_state_;
    bool terminal_;

    const CrossingStateParameters<Domain>& parameters_;
};



#endif //CROSSING_STATE_H
