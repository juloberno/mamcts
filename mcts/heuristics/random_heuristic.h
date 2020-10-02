// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef RANDOM_HEURISTIC_H
#define RANDOM_HEURISTIC_H

#include "mcts/mcts.h"
#include <iostream>
#include <chrono>

 namespace mcts {
// assumes all agents have equal number of actions and the same node statistic
class RandomHeuristic :  public mcts::Heuristic<RandomHeuristic>, mcts::RandomGenerator
{
public:
    RandomHeuristic(const MctsParameters& mcts_parameters) :
            mcts::Heuristic<RandomHeuristic>(mcts_parameters),
            RandomGenerator(mcts_parameters.RANDOM_SEED) {}

    template<class S, class SE, class SO, class H>
    std::pair<SE, std::unordered_map<AgentIdx, SO>> calculate_heuristic_values(const std::shared_ptr<StageNode<S,SE,SO,H>> &node) {
        //catch case where newly expanded state is terminal
        if(node->get_state()->is_terminal()){
            const auto ego_agent_idx = node->get_state()->get_ego_agent_idx();
            const ActionIdx num_ego_actions = node->get_state()->get_num_actions(ego_agent_idx); 
            SE ego_heuristic(num_ego_actions, node->get_state()->get_ego_agent_idx(), mcts_parameters_);
            ego_heuristic.set_heuristic_estimate(0.0f, {0.0f, 0.0f});
            std::unordered_map<AgentIdx, SO> other_heuristic_estimates;
            for (const auto& ai : node->get_state()->get_other_agent_idx())
            { 
              SO statistic(node->get_state()->get_num_actions(ai), ai, mcts_parameters_);
              statistic.set_heuristic_estimate(0.0f, {0.0f, 0.0f});
              other_heuristic_estimates.insert(std::pair<AgentIdx, SO>(ai, statistic));
            }
            return std::pair<SE, std::unordered_map<AgentIdx, SO>>(ego_heuristic, other_heuristic_estimates) ;
        }
        
        namespace chr = std::chrono;
        auto start = std::chrono::high_resolution_clock::now();
        std::shared_ptr<S> state = node->get_state()->clone();

        Reward ego_accum_reward = 0.0f;
        std::unordered_map<AgentIdx, Reward> other_accum_rewards;
        for (const auto& ai : state->get_other_agent_idx()) {
          other_accum_rewards[ai] = 0.0f;
        }

        EgoCosts accum_cost = {0.0f, 0.0f};
        const double k_discount_factor = mcts_parameters_.DISCOUNT_FACTOR; 
        double modified_discount_factor = k_discount_factor;
        int num_iterations = 0;
        auto current_depth = node->get_depth();
        
        while((!state->is_terminal())&&(num_iterations<mcts_parameters_.random_heuristic.MAX_NUMBER_OF_ITERATIONS)&&
                (std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count() 
                    < mcts_parameters_.random_heuristic.MAX_SEARCH_TIME ) &&
                  current_depth <= mcts_parameters_.MAX_SEARCH_DEPTH) {
            // Build joint action by calling statistics for each agent
            JointAction jointaction(state->get_num_agents());
            SE ego_statistic(state->get_num_actions(state->get_ego_agent_idx()),
                          state->get_ego_agent_idx(),
                          mcts_parameters_);
            jointaction[S::ego_agent_idx] = ego_statistic.choose_next_action(*state);
            AgentIdx action_idx = 1;
            for (const auto& ai : state->get_other_agent_idx()) {
              SO statistic(state->get_num_actions(ai), ai, mcts_parameters_);
              jointaction[action_idx] = statistic.choose_next_action(*state);
              action_idx++;
            }

            EgoCosts ego_cost{0.0f};
            std::vector<Reward> step_rewards(state->get_num_agents());
            auto new_state = state->execute(jointaction, step_rewards, ego_cost);

            ego_accum_reward += modified_discount_factor*step_rewards[S::ego_agent_idx];
            AgentIdx reward_idx = 1;
            for (const auto& ai : state->get_other_agent_idx()) {
              other_accum_rewards[ai] = modified_discount_factor*step_rewards[reward_idx];
              action_idx++;
            }

            accum_cost[0] += modified_discount_factor*ego_cost[0];
            modified_discount_factor = modified_discount_factor*k_discount_factor;

            state = new_state->clone();
            num_iterations +=1;
            current_depth += 1;
         };
        // generate an extra node statistic for each agent
        SE ego_heuristic(0, node->get_state()->get_ego_agent_idx(), mcts_parameters_);
        const auto ego_agent_idx = node->get_state()->get_ego_agent_idx();
        const ActionIdx num_ego_actions = node->get_state()->get_num_actions(ego_agent_idx);
        ego_heuristic.set_heuristic_estimate(ego_accum_reward, 
            EgoCosts({double(accum_cost[0] > 0.0)*std::pow(1.0/num_ego_actions, num_iterations), 0.0f})); // correct by probability of random selected ego actions
        std::unordered_map<AgentIdx, SO> other_heuristic_estimates;
        AgentIdx reward_idx=1;
        for (auto agent_idx : node->get_state()->get_other_agent_idx())
        {
            SO statistic(0, agent_idx, mcts_parameters_);
            statistic.set_heuristic_estimate(other_accum_rewards[agent_idx], accum_cost);
            other_heuristic_estimates.insert(std::pair<AgentIdx, SO>(agent_idx, statistic));
            reward_idx++;
        }
        return std::pair<SE, std::unordered_map<AgentIdx, SO>>(ego_heuristic, other_heuristic_estimates);
    }

};

 } // namespace mcts

#endif