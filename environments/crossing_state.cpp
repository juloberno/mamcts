// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "mcts/mcts.h"
#include "environments/crossing_state.h"


int CrossingStateParameters::MAX_VELOCITY_OTHER = 3;
int CrossingStateParameters::MIN_VELOCITY_OTHER = -3;
int CrossingStateParameters::MAX_VELOCITY_EGO = 2;
int CrossingStateParameters::MIN_VELOCITY_EGO = -1;
int CrossingStateParameters::CHAIN_LENGTH = 21; /* 10 is crossing point (21-1)/2+1 */
int CrossingStateParameters::EGO_GOAL_POS = 12;
