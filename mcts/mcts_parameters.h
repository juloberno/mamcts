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
  unsigned int RANDOM_SEED;

  struct RandomHeuristicParameters {
      double MAX_SEARCH_TIME;
      unsigned int MAX_NUMBER_OF_ITERATIONS;
  };

  struct UctStatisticParameters {
      double LOWER_BOUND;
      double UPPER_BOUND;
      double EXPLORATION_CONSTANT;
  };

  struct HypothesisStatisticParameters {
      bool COST_BASED_ACTION_SELECTION;
      double UPPER_COST_BOUND;
      double LOWER_COST_BOUND;
      double PROGRESSIVE_WIDENING_K;
      double PROGRESSIVE_WIDENING_ALPHA;
      double EXPLORATION_CONSTANT;
  };

  struct HypothesisBeliefTrackerParameters {
      unsigned int RANDOM_SEED_HYPOTHESIS_SAMPLING;
      unsigned int HISTORY_LENGTH;
      float PROBABILITY_DISCOUNT;
      int POSTERIOR_TYPE;
  };

  HypothesisStatisticParameters hypothesis_statistic;
  UctStatisticParameters uct_statistic;
  RandomHeuristicParameters random_heuristic;
  HypothesisBeliefTrackerParameters hypothesis_belief_tracker;
};


inline MctsParameters mcts_default_parameters() {
  MctsParameters parameters;
  parameters.DISCOUNT_FACTOR = 0.9;
  parameters.RANDOM_SEED = 1000;
  
  parameters.random_heuristic.MAX_SEARCH_TIME = 10;
  parameters.random_heuristic.MAX_NUMBER_OF_ITERATIONS = 1000;

  parameters.uct_statistic.LOWER_BOUND = -1000;
  parameters.uct_statistic.UPPER_BOUND = 100;
  parameters.uct_statistic.EXPLORATION_CONSTANT = 0.7;

  parameters.hypothesis_statistic.COST_BASED_ACTION_SELECTION = false;
  parameters.hypothesis_statistic.LOWER_COST_BOUND = 0;
  parameters.hypothesis_statistic.UPPER_COST_BOUND = 1;
  parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_ALPHA = 0.5;
  parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_K = 1;
  parameters.hypothesis_statistic.EXPLORATION_CONSTANT = 0.7;

  parameters.hypothesis_belief_tracker.RANDOM_SEED_HYPOTHESIS_SAMPLING = 1000;
  parameters.hypothesis_belief_tracker.HISTORY_LENGTH = 4;
  parameters.hypothesis_belief_tracker.PROBABILITY_DISCOUNT = 1.0f;
  parameters.hypothesis_belief_tracker.POSTERIOR_TYPE = 0; // = HypothesisBeliefTracker::PRODUCT;

  return parameters;
}
} // namespace mcts


#endif 
