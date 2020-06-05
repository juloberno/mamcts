// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef RANDOM_ACTIONS_H
#define RANDOM_ACTIONS_H

#include "mcts/mcts.h"
#include <iostream>
#include <iomanip>

namespace mcts {

// A upper confidence bound implementation
class RandomActions : public mcts::NodeStatistic<RandomActions>, mcts::RandomGenerator
{
public:
    MCTS_TEST
    FRIEND_COST_CONSTRAINED_STATISTIC

    RandomActions(ActionIdx num_actions, AgentIdx agent_idx, const MctsParameters & mcts_parameters) :
             NodeStatistic<RandomActions>(num_actions, agent_idx, mcts_parameters),
             RandomGenerator(mcts_parameters.RANDOM_SEED) {}

    ~RandomActions() {};

    template <class S>
    ActionIdx choose_next_action(const S& state) {
        // Select randomly an unexpanded action
        std::uniform_int_distribution<ActionIdx> random_action_selection(0, num_actions_-1);
        ActionIdx action = random_action_selection(random_generator_);
        return action;
    }

    ActionIdx get_best_action() const {
        throw std::logic_error("Not making sense for this implementation");
    }

    void update_from_heuristic(const NodeStatistic<RandomActions>& heuristic_statistic) {
    }

    void update_statistic(const NodeStatistic<RandomActions>& changed_child_statistic) {
    }

    void set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost)
    {}

    std::string print_node_information() const {return "";}

    std::string print_edge_information(const ActionIdx& action ) const {return "";}

};

} // namespace mcts

#endif