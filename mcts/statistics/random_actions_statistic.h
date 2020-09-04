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
class RandomActionsStatistic : public mcts::NodeStatistic<RandomActionsStatistic>, mcts::RandomGenerator
{
public:
    MCTS_TEST
    FRIEND_COST_CONSTRAINED_STATISTIC

    RandomActionsStatistic(ActionIdx num_actions, AgentIdx agent_idx, const MctsParameters & mcts_parameters) :
             NodeStatistic<RandomActionsStatistic>(num_actions, agent_idx, mcts_parameters),
             RandomGenerator(mcts_parameters.RANDOM_SEED),
             num_expanded_actions_(0),
             expanded_actions_counts(),
             total_node_visits_(0),
             progressive_widening_k(mcts_parameters.random_actions_statistic.PROGRESSIVE_WIDENING_K),
             progressive_widening_alpha(mcts_parameters.random_actions_statistic.PROGRESSIVE_WIDENING_ALPHA) {}

    ~RandomActionsStatistic() {};

    template <class S>
    ActionIdx choose_next_action(const S& state) {
        // Select randomly an unexpanded action
        if(require_progressive_widening()) {
          while(true) { 
            std::uniform_int_distribution<ActionIdx> random_action_selection(0, num_actions_-1);
            ActionIdx action = random_action_selection(random_generator_);
            const auto it = expanded_actions_counts.find(action);
            if(it == expanded_actions_counts.end()) {
              num_expanded_actions_++;
              expanded_actions_counts[action]++;
              return action;
            }
          }
        } else {
          ActionIdx lowest_action = expanded_actions_counts.begin()->first;
          for ( const auto action_count : expanded_actions_counts) {
              if(expanded_actions_counts.at(lowest_action) > action_count.second) {
                lowest_action = action_count.first;
              }
          }
          expanded_actions_counts[lowest_action]++;
          return lowest_action;
        }
    }

    ActionIdx get_best_action() const {
        throw std::logic_error("Not making sense for this implementation");
    }

    void update_from_heuristic(const NodeStatistic<RandomActionsStatistic>& heuristic_statistic) {
      total_node_visits_++;
    }

    void update_statistic(const NodeStatistic<RandomActionsStatistic>& changed_child_statistic) {
      total_node_visits_++;
    }

    void set_heuristic_estimate(const Reward& accum_rewards, const Cost& accum_ego_cost)
    {}

    std::string sprintf() const {return "";}

    std::string print_node_information() const {return "";}

    std::string print_edge_information(const ActionIdx& action ) const {return "";}

    inline bool require_progressive_widening() const {
        const auto widening_term = progressive_widening_k * std::pow(total_node_visits_,
                progressive_widening_alpha);
        return (num_expanded_actions_ <= widening_term && num_expanded_actions_ < num_actions_);
    }

private:
  std::unordered_map<ActionIdx, std::size_t> expanded_actions_counts;
  unsigned int num_expanded_actions_;
  unsigned int total_node_visits_;

  const double progressive_widening_k;
  const double progressive_widening_alpha;

};

} // namespace mcts

#endif