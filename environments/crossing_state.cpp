// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "mcts/mcts.h"
#include "environments/crossing_state.h"
#include "environments/viewer.hpp"


int CrossingStateParameters::MAX_VELOCITY_OTHER = 3;
int CrossingStateParameters::MIN_VELOCITY_OTHER = -3;
int CrossingStateParameters::MAX_VELOCITY_EGO = 2;
int CrossingStateParameters::MIN_VELOCITY_EGO = -1;
int CrossingStateParameters::CHAIN_LENGTH = 21; /* 10 is crossing point (21-1)/2+1 */
int CrossingStateParameters::EGO_GOAL_POS = 12;


void CrossingState::draw(mcts::Viewer* viewer) const {
    // draw map ( crossing point is always at zero)
    const float state_draw_dst = 10.0f;
    const float linewidth = 0.5;
    const float state_draw_size = 2;
    const float factor_draw_current_state = 1.5;

    // draw lines equally spaced angles with small points
    // indicating states and larger points indicating the current state
    const float angle_delta = M_PI/(num_other_agents+2); // one for ego 
    const float line_radius = state_draw_dst*CrossingStateParameters::CHAIN_LENGTH/2.0f;
    for(int i = 0; i < num_other_agents+1; ++i) {
        float start_angle = 1.5*M_PI - (i+1)*angle_delta;
        float end_angle = start_angle + M_PI;
        std::pair<float, float> line_start{cos(start_angle)*line_radius, sin(start_angle)*line_radius};
        std::pair<float, float> line_end{cos(end_angle)*line_radius, sin(end_angle)*line_radius};
        std::tuple<float,float,float,float> color{0,0,0,0};

        viewer->drawLine(line_start, line_end,
                    linewidth, color);

        // Draw current states
        AgentState state;
        if(i == std::floor(num_other_agents/2)) {
            state = ego_state_;
            color = {0.8,0,0,0}; 
        } else {
            const AgentIdx agt_idx = i;
            if (i > std::floor(num_other_agents/2)) {
            const AgentIdx agt_idx = i-1;
            }
            state = other_agent_states_[agt_idx];
        }

        for (int y = 0; y < CrossingStateParameters::CHAIN_LENGTH; ++y) {
            const auto px = line_start.first + (line_end.first - line_start.first) * static_cast<float>(state.x_pos / CrossingStateParameters::CHAIN_LENGTH);
            const auto py = line_start.second + (line_end.second - line_start.second) * static_cast<float>(state.x_pos / CrossingStateParameters::CHAIN_LENGTH);
            float pointsize_temp = state_draw_size; 
            if (state.x_pos == y) {
                pointsize_temp *= factor_draw_current_state;
            }
            viewer->drawPoint(px, py,
                        pointsize_temp, color);
        }
    }

}