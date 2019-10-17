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
        static double RANDOM_GENERATOR_SEED;
        static double MAX_NUMBER_OF_ITERATIONS;

        struct RandomHeuristic {
            static double MAX_SEARCH_TIME;
            static double MAX_NUMBER_OF_ITERATIONS;
        };

        struct UctStatistic {
            static double LOWER_BOUND;
            static double UPPER_BOUND;
            static double EXPLORATION_CONSTANT;
        };

        struct HypothesisStatistic {
            static bool COST_BASED_ACTION_SELECTION;
            static double UPPER_COST_BOUND;
            static double LOWER_COST_BOUND;
            static double PROGRESSIVE_WIDENING_K;
            static double PROGRESSIVE_WIDENING_ALPHA;
            static double EXPLORATION_CONSTANT;
        };
    };
} // namespace mcts


#endif 
