// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef UCT_COST_CONSTRAINED_STATISTIC_H
#define UCT_COST_CONSTRAINED_STATISTIC_H

#include "mcts/statistics/uct_statistic.h
#include <iostream>
#include <iomanip>

namespace mcts {

// A upper confidence bound implementation
class CostConstrainedStatistic : public mcts::NodeStatistic<CostConstrainedStatistic>, mcts::RandomGenerator
{
public:
    MCTS_TEST

    CostConstrainedStatistic(ActionIdx num_actions, AgentIdx agent_idx, const MctsParameters & mcts_parameters) :
             NodeStatistic<CostConstrainedStatistic>(num_actions, agent_idx, mcts_parameters),
             RandomGenerator(mcts_parameters.RANDOM_SEED) 
             {}

    ~CostConstrainedStatistic() {};

    template <class S>
    ActionIdx choose_next_action(const S& state) {
    }

    ActionIdx get_best_action() {
    }

    void update_from_heuristic(const NodeStatistic<CostConstrainedStatistic>& heuristic_statistic)
    {
      const CostConstrainedStatistic& heuristic_statistic_impl = heuristic_statistic.impl();

      auto& uct_reward_statistics = heuristic_statistic_impl.reward_statistic_.uct_statistics_;
      const auto heuristic_reward_value = heuristic_statistic_impl.reward_statistic_.value_;
      reward_statistic_.update_from_heuristic_from_backpropagated_value(uct_reward_statistics, heuristic_reward_value);

      auto& uct_cost_statistics = heuristic_statistic_impl.cost_statistic_.uct_statistics_;
      const auto heuristic_cost_value = heuristic_statistic_impl.cost_statistic_.value_;
      cost_statistic_.update_from_heuristic_from_backpropagated_value(uct_cost_statistics, heuristic_cost_value);
    }

    void update_statistic(const NodeStatistic<CostConstrainedStatistic>& changed_child_statistic) {

    }

    void set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost)
    {
      todo wrong 
       reward_statistic_.set_heuristic_estimate(accum_rewards);
       cost_statistic_.set_heuristic_estimate(accum_ego_cost);
    }

    std::string print_node_information() const
    {
    }

    std::string print_edge_information(const ActionIdx& action ) const
    {
    }

private:

    UctStatistic reward_statistic_;
    UctStatistic cost_statistic_;
    

};

} // namespace mcts

#endif