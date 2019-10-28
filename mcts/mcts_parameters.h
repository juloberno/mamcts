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
        double DISCOUNT_FACTOR;

        struct RandomHeuristic {
            double MAX_SEARCH_TIME;
            unsigned int MAX_NUMBER_OF_ITERATIONS;
        };

        struct UctStatistic {
            double LOWER_BOUND;
            double UPPER_BOUND;
            double EXPLORATION_CONSTANT;
        };

        struct HypothesisStatistic {
            bool COST_BASED_ACTION_SELECTION;
            double UPPER_COST_BOUND;
            double LOWER_COST_BOUND;
            double PROGRESSIVE_WIDENING_K;
            double PROGRESSIVE_WIDENING_ALPHA;
            double EXPLORATION_CONSTANT;
        };

        HypothesisStatistic hypothesis_statistic;
        UctStatistic uct_statistic;
        RandomHeuristic random_heuristic;
    };
} // namespace mcts


#endif 
