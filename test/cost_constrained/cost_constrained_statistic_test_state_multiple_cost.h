// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef COST_CONSTRAINED_TEST_STATE_MULTIPLE_COST_H
#define COST_CONSTRAINED_TEST_STATE_MULTIPLE_COST_H

#include <iostream>
#include <cmath>
#include <random>
#include "test/cost_constrained/cost_constrained_statistic_test_state_single_cost.h"

typedef double Probability;

using namespace mcts;

class CostConstrainedStatisticTestStateMultipleCost :
   public CostConstrainedStatisticTestStateBase<CostConstrainedStatisticTestStateMultipleCost> {
public:
    CostConstrainedStatisticTestStateMultipleCost(int current_state, int n_steps, Cost collision_risk1, Cost collision_risk2,
                                     Reward reward_goal1, Reward reward_goal2, bool is_terminal, unsigned int seed) : 
                          CostConstrainedStatisticTestStateBase<CostConstrainedStatisticTestStateMultipleCost>(
                            current_state, n_steps, collision_risk1, collision_risk2, reward_goal1,  reward_goal2,
                            is_terminal, seed) {}           
    std::vector<Cost> get_collision_cost() const {
       return {0.0f, 1.0f};
    }

    std::vector<Cost> get_goal1_cost() const {
       return {0.0f, 0.0f};
    }

    std::vector<Cost> get_goal2_cost() const {
       return {1.0f, 0.0f};
    }

    std::vector<Cost> get_other_cost() const {
       return {0.0f, 0.0f};
    }
};



#endif 
