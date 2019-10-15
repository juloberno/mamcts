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
    const float state_draw_dst = 1.0f;
    const float linewidth = 0.5;
    const float state_draw_size = 50;
    const float factor_draw_current_state = 1.5;

    // draw lines equally spaced angles with small points
    // indicating states and larger points indicating the current state
    const float angle_delta = M_PI/(num_other_agents+2); // one for ego 
    const float line_radius = state_draw_dst*(CrossingStateParameters::CHAIN_LENGTH-1)/2.0f;
    for(int i = 0; i < num_other_agents+1; ++i) {
        float start_angle = 1.5*M_PI - (i+1)*angle_delta;
        float end_angle = start_angle + M_PI;
        std::pair<float, float> line_x{cos(start_angle)*line_radius, cos(end_angle)*line_radius };
        std::pair<float, float> line_y{sin(start_angle)*line_radius, sin(end_angle)*line_radius};
        std::tuple<float,float,float,float> color{0,0,0,0};


        // Differentiate between ego and other agents
        AgentState state;
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
        for (int y = 0; y < CrossingStateParameters::CHAIN_LENGTH; ++y) {
            const auto px = line_x.first + (line_x.second - line_x.first) * static_cast<float>(y) /
                                                             static_cast<float>(CrossingStateParameters::CHAIN_LENGTH-1);
            const auto py = line_y.first + (line_y.second - line_y.first) * static_cast<float>(y) /
                                                             static_cast<float>(CrossingStateParameters::CHAIN_LENGTH-1);
            float pointsize_temp = state_draw_size; 
            if (state.x_pos == y) {
                pointsize_temp *= factor_draw_current_state;
            }
            viewer->drawPoint(px, py,
                        pointsize_temp, color);
        }
    }

}