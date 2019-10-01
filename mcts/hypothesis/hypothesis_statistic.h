// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_HYPOTHESIS_STATISTICS_H
#define MCTS_HYPOTHESIS_STATISTICS_H

#include "mcts/mcts.h"
#include "mcts/hypothesis/hypothesis_state.h"
#include <map>

namespace mcts {

class HypothesisStatistic : public mcts::NodeStatistic<HypothesisStatistic>,
                                   mcts::RandomGenerator,
                                   mcts::RequiresHypothesis
{
public:
    MCTS_TEST;

    HypothesisStatistic(ActionIdx num_actions, AgentIdx agent_idx) :
                     NodeStatistic<HypothesisStatistic>(num_actions, agent_idx) {}

    template <class S>
    ActionIdx choose_next_action(const StateInterface<S>& state, std::vector<int>& unexpanded_actions) {
                const HypothesisStateInterface<S>& impl = state.impl();
                return impl.plan_action_current_hypothesis(agent_idx_);
    }

    void update_statistic(const NodeStatistic<HypothesisStatistic>& changed_child_statistic) {}; 
    
    void update_from_heuristic(const NodeStatistic<HypothesisStatistic>& heuristic_statistic) {}; 
    
    ActionIdx get_best_action() {return 0;};

    void set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost) {};

    std::string print_node_information() const {return "";};

    std::string print_edge_information(const ActionIdx& action) const { return "";};


private:

};

} // namespace mcts


#endif // MCTS_HYPOTHESIS_STATISTICS_H