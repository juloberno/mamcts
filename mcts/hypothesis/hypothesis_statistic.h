// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_HYPOTHESIS_STATISTICS_H
#define MCTS_HYPOTHESIS_STATISTICS_H

#include "mcts/mcts.h"
#include "statistics/hypothesis.h"
#include <map>
#include "common.h"

namespace mcts {

class HypothesisStatistic : public mcts::NodeStatistic<HypothesisStatistic>, mcts::RandomGenerator
{
public:
    MCTS_TEST;

    HypothesisStatistic(ActionIdx num_actions, const Hypothesis* current_hypothesis_ptr) :
                     NodeStatistic<HypothesisStatistic>(num_actions),
                     current_hypothesis_ptr_(current_hypothesis_ptr) {}

    template <class S>
    ActionIdx choose_next_action(const StateInterface<S>& state, std::vector<int>& unexpanded_actions) 
            {return state_->plan_action_current_hypothesis(agent_idx_);}

    void update_statistic(const NodeStatistic<HypothesisStatistic>& changed_child_statistic); 
    
    void update_from_heuristic(const NodeStatistic<HypothesisStatistic>& heuristic_statistic); 
    
    ActionIdx get_best_action();

    void set_heuristic_estimate(const Reward& accum_rewards);

    void collect_reward(const Reward& reward, const ActionIdx& action_idx);

    std::string print_node_information() const;

    std::string print_edge_information(const ActionIdx& action) const;

private:
   

};

} // namespace mcts



#endif // MCTS_HYPOTHESIS_STATISTICS_H