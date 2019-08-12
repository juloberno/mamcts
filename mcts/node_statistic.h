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

namespace mcts {



template <class Implementation>
class NodeStatistic
{
public:
    MCTS_TEST;

    NodeStatistic(ActionIdx num_actions) : num_actions_(num_actions) {}
    template <class S>
    ActionIdx choose_next_action(const StateInterface<S>& state, std::vector<int>& unexpanded_actions);
    void update_statistic(const NodeStatistic<Implementation>& changed_child_statistic); // update statistic during backpropagation from child node
    void update_from_heuristic(const NodeStatistic<Implementation>& heuristic_statistic); // update statistic during backpropagation from heuristic estimate
    ActionIdx get_best_action();

    void set_heuristic_estimate(const Reward& accum_rewards);

    void collect_reward(const Reward& reward, const ActionIdx& action_idx);

    std::string print_node_information() const;
    std::string print_edge_information(const ActionIdx& action) const;

    Implementation& impl();
    const Implementation& impl() const;

protected:
    std::pair<ActionIdx, Reward> collected_reward_;
    ActionIdx num_actions_;

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
ActionIdx NodeStatistic<Implementation>::choose_next_action(const StateInterface<S>& state, std::vector<int>& unexpanded_actions)  {
    return impl().choose_next_action(state, unexpanded_actions);
}

template <class Implementation>
ActionIdx NodeStatistic<Implementation>::get_best_action()  {
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
void NodeStatistic<Implementation>::collect_reward(const mcts::Reward &reward, const ActionIdx& action_idx) {
    collected_reward_= std::pair<ActionIdx, Reward>(action_idx, reward);
}

template <class Implementation>
void NodeStatistic<Implementation>::set_heuristic_estimate(const Reward& accum_rewards) {
    return impl().set_heuristic_estimate(accum_rewards);
}


} // namespace mcts



#endif