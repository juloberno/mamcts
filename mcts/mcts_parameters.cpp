// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#include "mcts/mcts_parameters.h"


namespace mcts {

    double MctsParameters::DISCOUNT_FACTOR = 0.9;
    double MctsParameters::EXPLORATION_CONSTANT = 0.7;

    double MctsParameters::RANDOM_GENERATOR_SEED = 1000;
    double MctsParameters::MAX_SEARCH_TIME = 10000;
    double MctsParameters::MAX_NUMBER_OF_ITERATIONS = 1000;
    double MctsParameters::MAX_SEARCH_TIME_RANDOM_HEURISTIC = 50;
    double MctsParameters::MAX_NUMBER_OF_ITERATIONS_RANDOM_HEURISTIC = 100;

    double MctsParameters::LOWER_BOUND = -1010;
    double MctsParameters::UPPER_BOUND = 95;
    
} // namespace mcts
