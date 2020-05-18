// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef UCT_COST_CONSTRAINED_STATISTIC_H
#define UCT_COST_CONSTRAINED_STATISTIC_H

#include "mcts/statistics/uct_statistic.h"
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
             reward_statistic_(num_actions, agent_idx, mcts_parameters_),
             cost_statistic_(num_actions, agent_idx, mcts_parameters_),
             RandomGenerator(mcts_parameters.RANDOM_SEED) 
             {}

    ~CostConstrainedStatistic() {};

    template <class S>
    ActionIdx choose_next_action(const S& state) {
    }

    ActionIdx get_best_action() {
        // Here sampling of greedy policy
        // why is this non-const overall?
    }

    static double calculate_next_lambda() {
        // should be called only at root statistic 
        // computes gradient and does clipping, the it returns new lambda
        // should be made const
        // returned lambda is set in parameters for next iteration
    }

    void update_from_heuristic(const NodeStatistic<CostConstrainedStatistic>& heuristic_statistic)
    {
      const CostConstrainedStatistic& statistic_impl = heuristic_statistic.impl();

      const auto heuristic_reward_value = statistic_impl.reward_statistic_.value_;
      reward_statistic_.update_from_heuristic_from_backpropagated(heuristic_reward_value);

      const auto heuristic_cost_value = statistic_impl.cost_statistic_.value_;
      cost_statistic_.update_from_heuristic_from_backpropagated(heuristic_cost_value);
    }

    void update_statistic(const NodeStatistic<CostConstrainedStatistic>& changed_child_statistic) {
      const CostConstrainedStatistic& statistic_impl = changed_child_statistic.impl();

      const auto reward_latest_return = statistic_impl.reward_statistic_.latest_return_;
      reward_statistic_.update_statistics_from_backpropagated(reward_latest_return);

      const auto cost_latest_return = statistic_impl.cost_statistic_.latest_return_;
      cost_statistic_.update_statistics_from_backpropagated(cost_latest_return);
    }

    void set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost)
    {
       reward_statistic_.set_heuristic_estimate_from_backpropagated(accum_rewards);
       cost_statistic_.set_heuristic_estimate_from_backpropagated(accum_ego_cost);
    }

    std::string print_node_information() const
    {
        return "";
    }

    std::string print_edge_information(const ActionIdx& action ) const
    {
        return "";
    }

private:

    UctStatistic reward_statistic_;
    UctStatistic cost_statistic_;
    

};

} // namespace mcts

#endif