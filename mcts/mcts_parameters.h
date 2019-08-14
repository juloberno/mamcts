// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_PARAMETERS_H
#define MCTS_PARAMETERS_H


namespace mcts{

    struct MctsParameters{

        //MCTS
        static double DISCOUNT_FACTOR;
        static double EXPLORATION_CONSTANT;

        static double RANDOM_GENERATOR_SEED;
        static double MAX_NUMBER_OF_ITERATIONS;
        static double MAX_SEARCH_TIME_RANDOM_HEURISTIC;
        static double MAX_NUMBER_OF_ITERATIONS_RANDOM_HEURISTIC;

        static double LOWER_BOUND;
        static double UPPER_BOUND;
    };
} // namespace mcts


#endif 
