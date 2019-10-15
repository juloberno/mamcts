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
    // --- ego
    viewer->drawLine({0, 0}, {0, state_draw_dst*CrossingStateParameters::CHAIN_LENGTH},
                    linewidth, {0,0,0,0});

    // draw lines equally spaced angles
    const float angle_delta = M_PI/(num_other_agents+1); 
    for(int i = 0; i < num_other_agents; ++i) {
        std::pair<float, float> line_start{};
        std::pair<float, float> line_end{};
    }


    // draw other agents
    for(const auto& agent_state : other_agent_states_) {

    }


}