// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef RANDOM_HEURISTIC_H
#define RANDOM_HEURISTIC_H

#include "mcts/mcts.h"
#include "mcts/mcts_parameters.h"
#include <iostream>
#include <chrono>

using namespace mcts;
// assumes all agents have equal number of actions and the same node statistic
class RandomHeuristic :  public mcts::Heuristic<RandomHeuristic>, mcts::RandomGenerator
{
public:
    template<class S, class SE, class SO, class H>
    std::vector<SE> get_heuristic_values(const std::shared_ptr<StageNode<S,SE,SO,H>> &node) {
        //catch case where newly expanded state is terminal
        if(node->get_state()->is_terminal()){
            const AgentIdx num_agents = node->get_state()->get_agent_idx().size();
            const ActionIdx num_actions = node->get_state()->get_num_actions(S::ego_agent_idx); 
            std::vector<SE> statistics(num_agents,SE(num_actions));
            for (AgentIdx ai = 0; ai < num_agents; ++ai){
                statistics.at(ai).set_heuristic_estimate(0);
            }
            return  statistics;
        }
        
        namespace chr = std::chrono;
        auto start = std::chrono::high_resolution_clock::now();
        std::shared_ptr<S> state = node->get_state()->clone();
        const AgentIdx num_agents = node->get_state()->get_agent_idx().size();
        const ActionIdx num_actions = node->get_state()->get_num_actions(S::ego_agent_idx);

        std::vector<Reward> accum_rewards(num_agents);
        std::vector<Reward> step_rewards(state->get_agent_idx().size());
        const double k_discount_factor = mcts::MctsParameters::DISCOUNT_FACTOR; 
        double modified_discount_factor = k_discount_factor;
        int num_iterations = 0;
        while((!state->is_terminal())&&(num_iterations<max_iterations_random_heuristic)&&(std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::high_resolution_clock::now() - start ).count() < max_search_time_random_heuristic ))
        {
             auto new_state = state->execute(random_joint_action(num_actions, num_agents), step_rewards);
             state = new_state->clone();
             num_iterations +=1;
             // discount the rewards of the current step
             for(uint i=0; i<step_rewards.size(); i++){
                 step_rewards.at(i) = step_rewards.at(i)*modified_discount_factor;
             }
             accum_rewards += step_rewards;
             modified_discount_factor = modified_discount_factor*k_discount_factor;

         };
        // generate an extra node statistic for each agent
        std::vector<SE> statistics(num_agents,SE(num_actions));
        for (AgentIdx ai = 0; ai < num_agents; ++ai)
        {
            statistics.at(ai).set_heuristic_estimate(accum_rewards.at(ai));
         }

        return  statistics;
    }

    JointAction random_joint_action(ActionIdx num_actions, AgentIdx num_agents)
    {
        std::uniform_int_distribution<ActionIdx> random_action_selection(0,num_actions-1);

        auto gen = [&](){
            return random_action_selection(random_generator_);
        };

        JointAction ja(num_agents);
        std::generate(std::begin(ja), std::end(ja), gen);
        return ja;
    }
    const double max_search_time_random_heuristic = mcts::MctsParameters::MAX_SEARCH_TIME_RANDOM_HEURISTIC;
    const double max_iterations_random_heuristic = mcts::MctsParameters::MAX_NUMBER_OF_ITERATIONS_RANDOM_HEURISTIC;


};

#endif