// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_STATISTICS_H
#define MCTS_STATISTICS_H

#include "state.h"
#include <map>
#include "common.h"
#include "mcts_parameters.h"

namespace mcts {

template <class Implementation>
class NodeStatistic
{
public:
    MCTS_TEST;

    NodeStatistic(ActionIdx num_actions,
                 AgentIdx agent_idx, 
                 const MctsParameters& mcts_parameters) : 
                 num_actions_(num_actions),
                 agent_idx_(agent_idx),
                 mcts_parameters_(mcts_parameters) {}
    template <class S>
    ActionIdx choose_next_action(const StateInterface<S>& state);
    void update_statistic(const NodeStatistic<Implementation>& changed_child_statistic); // update statistic during backpropagation from child node
    void update_from_heuristic(const NodeStatistic<Implementation>& heuristic_statistic); // update statistic during backpropagation from heuristic estimate
    ActionIdx get_best_action() const;

    void set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost);

    void collect(const Reward& reward,  const Cost& cost, const ActionIdx& action_idx);

    std::string print_node_information() const;
    std::string print_edge_information(const ActionIdx& action) const;

    static void update_statistic_parameters(MctsParameters& parameters,
                                            const Implementation& root_statistic,
                                            const unsigned int& current_iteration);

    Implementation& impl();
    const Implementation& impl() const;

protected:
    std::pair<ActionIdx, Reward> collected_reward_;
    std::pair<ActionIdx, Cost> collected_cost_;
    ActionIdx num_actions_;
    AgentIdx agent_idx_;
    const MctsParameters& mcts_parameters_;
};

template <class Implementation>
Implementation& NodeStatistic<Implementation>::impl() {
    return *static_cast<Implementation*>(this);
}


template <class Implementation>
const Implementation& NodeStatistic<Implementation>::impl() const {
    return *static_cast<const Implementation*>(this);
}

template <class Implementation>
template <class S>
ActionIdx NodeStatistic<Implementation>::choose_next_action(const StateInterface<S>& state)  {
    return impl().choose_next_action(state);
}

template <class Implementation>
ActionIdx NodeStatistic<Implementation>::get_best_action() const {
    return impl().get_best_action();
}

template <class Implementation>
std::string NodeStatistic<Implementation>::print_node_information() const {
    return impl().print_node_information();
}

template <class Implementation>
std::string NodeStatistic<Implementation>::print_edge_information(const ActionIdx& action) const {
    return impl().print_edge_information();
}

template <class Implementation>
void NodeStatistic<Implementation>::update_statistic(const NodeStatistic<Implementation> &changed_child_statistic) {
    return impl().update_statistic(changed_child_statistic);
}

template <class Implementation>
void NodeStatistic<Implementation>::update_from_heuristic(const NodeStatistic<Implementation>& heuristic_statistic) {
    return impl().update_from_heuristic(heuristic_statistic);
}

template <class Implementation>
void NodeStatistic<Implementation>::collect(const mcts::Reward &reward, const mcts::Cost& cost, const ActionIdx& action_idx) {
    collected_reward_= std::pair<ActionIdx, Reward>(action_idx, reward);
    collected_cost_= std::pair<ActionIdx, Reward>(action_idx, cost);
}

template <class Implementation>
void NodeStatistic<Implementation>::set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost) {
    return impl().set_heuristic_estimate(accum_rewards, accum_ego_cost);
}

template <class Implementation>
void NodeStatistic<Implementation>::update_statistic_parameters(MctsParameters& parameters,
                                            const Implementation& root_statistic,
                                            const unsigned int& current_iteration) {
  return;
}


} // namespace mcts



#endif