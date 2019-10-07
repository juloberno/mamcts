// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "mcts/mcts_parameters.h"


namespace mcts {

    double MctsParameters::DISCOUNT_FACTOR = 0.9;

    double MctsParameters::RandomHeuristic::MAX_SEARCH_TIME = 1;
    double MctsParameters::RandomHeuristic::MAX_NUMBER_OF_ITERATIONS = 1000;

    double MctsParameters::UctStatistic::LOWER_BOUND = -1010;
    double MctsParameters::UctStatistic::UPPER_BOUND = 95;
    double MctsParameters::UctStatistic::EXPLORATION_CONSTANT = 0.7;

    bool MctsParameters::HypothesisStatistic::COST_BASED_ACTION_SELECTION = false;
    double MctsParameters::HypothesisStatistic::LOWER_COST_BOUND = 0;
    double MctsParameters::HypothesisStatistic::UPPER_COST_BOUND = 1;
    double MctsParameters::HypothesisStatistic::PROGRESSIVE_WIDENING_ALPHA = 0.5;
    double MctsParameters::HypothesisStatistic::PROGRESSIVE_WIDENING_K = 1;
    double MctsParameters::HypothesisStatistic::EXPLORATION_CONSTANT = 0.7;
} // namespace mcts
