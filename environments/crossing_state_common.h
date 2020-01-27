// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================


#ifndef MCTS_CROSSING_STATE_COMMON_H_
#define MCTS_CROSSING_STATE_COMMON_H_

#include "environments/crossing_state_parameters.h"
#include "mcts/hypothesis/hypothesis_state.h"

namespace mcts {

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
    AgentState() : x_pos(5), last_action(2.0f) {}
    AgentState(const Domain& x, const Domain& last_action) : 
            x_pos(x), last_action(last_action) {}
    Domain x_pos;
    Domain last_action;
};

} // namespace mcts

#endif