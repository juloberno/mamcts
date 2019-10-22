// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "mcts/mcts.h"
#include "environments/crossing_state.h"

template <>
int CrossingStateParameters<int>::MAX_VELOCITY_OTHER = 3;
template <>
int CrossingStateParameters<int>::MIN_VELOCITY_OTHER = -3;
template <>
int CrossingStateParameters<int>::MAX_VELOCITY_EGO = 2;
template <>
int CrossingStateParameters<int>::MIN_VELOCITY_EGO = -1;
template <>
int CrossingStateParameters<int>::CHAIN_LENGTH = 21; /* 10 is crossing point (21-1)/2+1 */
template <>
int CrossingStateParameters<int>::EGO_GOAL_POS = 12;

template <>
float CrossingStateParameters<float>::MAX_VELOCITY_OTHER = 3;
template <>
float CrossingStateParameters<float>::MIN_VELOCITY_OTHER = -3;
template <>
float CrossingStateParameters<float>::MAX_VELOCITY_EGO = 2;
template <>
float CrossingStateParameters<float>::MIN_VELOCITY_EGO = -1;
template <>
float CrossingStateParameters<float>::CHAIN_LENGTH = 21; /* 10 is crossing point (21-1)/2+1 */
template <>
float CrossingStateParameters<float>::EGO_GOAL_POS = 12;

