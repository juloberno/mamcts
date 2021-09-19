// Copyright (c) 2019 Julian Bernhard
// 
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.
// ========================================================

#ifndef MCTS_PARAMETERS_H
#define MCTS_PARAMETERS_H


#include <unordered_map>
#include <vector>


namespace mcts{


struct MctsParameters{
  //MCTS
  double DISCOUNT_FACTOR;
  unsigned int RANDOM_SEED;
  unsigned int MAX_NUMBER_OF_ITERATIONS;
  unsigned int MAX_SEARCH_TIME;
  unsigned int MAX_SEARCH_DEPTH;
  unsigned int MAX_NUMBER_OF_NODES;
  bool USE_BOUND_ESTIMATION;
  unsigned int NUM_PARALLEL_MCTS;
  bool USE_MULTI_THREADING;

  struct RandomHeuristicParameters {
      double MAX_SEARCH_TIME;
      unsigned int MAX_NUMBER_OF_ITERATIONS;
  };

  struct UctStatisticParameters {
      double LOWER_BOUND;
      double UPPER_BOUND;
      double EXPLORATION_CONSTANT;
      double PROGRESSIVE_WIDENING_K;
      double PROGRESSIVE_WIDENING_ALPHA;
  };

  struct CostConstrainedStatisticParameters {
      std::vector<double> LAMBDAS;
      double COST_UPPER_BOUND;
      double COST_LOWER_BOUND;
      double REWARD_UPPER_BOUND;
      double REWARD_LOWER_BOUND;
      std::vector<double> COST_CONSTRAINTS;
      double EXPLORATION_REDUCTION_FACTOR;
      double EXPLORATION_REDUCTION_CONSTANT_OFFSET;
      double EXPLORATION_REDUCTION_INIT;
      unsigned MIN_VISITS_POLICY_READY;
      double KAPPA;
      double GRADIENT_UPDATE_STEP;
      double TAU_GRADIENT_CLIP;
      double ACTION_FILTER_FACTOR;
      std::vector<bool> USE_COST_THRESHOLDING;
      std::vector<bool> USE_CHANCE_CONSTRAINED_UPDATES;
      std::vector<double> COST_THRESHOLDS;
      bool USE_LAMBDA_POLICY;
  };

  struct RandomActionsStatisticParameters {
      double PROGRESSIVE_WIDENING_K;
      double PROGRESSIVE_WIDENING_ALPHA;
  };

  struct HypothesisStatisticParameters {
      bool COST_BASED_ACTION_SELECTION;
      bool PROGRESSIVE_WIDENING_HYPOTHESIS_BASED;
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
      std::unordered_map<unsigned int, unsigned int> FIXED_HYPOTHESIS_SET;
  };

  HypothesisStatisticParameters hypothesis_statistic;
  UctStatisticParameters uct_statistic;
  RandomHeuristicParameters random_heuristic;
  RandomActionsStatisticParameters random_actions_statistic;
  HypothesisBeliefTrackerParameters hypothesis_belief_tracker;
  CostConstrainedStatisticParameters cost_constrained_statistic;
};


inline MctsParameters mcts_default_parameters() {
  MctsParameters parameters;
  parameters.DISCOUNT_FACTOR = 0.9;
  parameters.RANDOM_SEED = 1000;
  parameters.MAX_NUMBER_OF_ITERATIONS = 10000;
  parameters.MAX_NUMBER_OF_NODES = 10000;
  parameters.MAX_SEARCH_TIME = 1000;
  parameters.MAX_SEARCH_DEPTH = 1000;
  parameters.USE_BOUND_ESTIMATION = true;
  parameters.NUM_PARALLEL_MCTS = 1;
  parameters.USE_MULTI_THREADING = false;
  
  parameters.random_heuristic.MAX_SEARCH_TIME = 10;
  parameters.random_heuristic.MAX_NUMBER_OF_ITERATIONS = 1000;

  parameters.uct_statistic.LOWER_BOUND = -1000;
  parameters.uct_statistic.UPPER_BOUND = 100;
  parameters.uct_statistic.EXPLORATION_CONSTANT = 0.7;
  parameters.uct_statistic.PROGRESSIVE_WIDENING_K = 4.0; //< defaults to no prog. widening
  parameters.uct_statistic.PROGRESSIVE_WIDENING_ALPHA = 0.25; //< defaults to no prog. widening

  parameters.hypothesis_statistic.COST_BASED_ACTION_SELECTION = false;
  parameters.hypothesis_statistic.LOWER_COST_BOUND = 0;
  parameters.hypothesis_statistic.UPPER_COST_BOUND = 100;
  parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_HYPOTHESIS_BASED = true;
  parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_ALPHA = 0.5;
  parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_K = 1.0;
  parameters.hypothesis_statistic.EXPLORATION_CONSTANT = 0.7;

  parameters.hypothesis_belief_tracker.RANDOM_SEED_HYPOTHESIS_SAMPLING = 1000;
  parameters.hypothesis_belief_tracker.HISTORY_LENGTH = 4;
  parameters.hypothesis_belief_tracker.PROBABILITY_DISCOUNT = 1.0f;
  parameters.hypothesis_belief_tracker.POSTERIOR_TYPE = 0; // = HypothesisBeliefTracker::PRODUCT;
  parameters.hypothesis_belief_tracker.FIXED_HYPOTHESIS_SET = {};

  return parameters;
}
} // namespace mcts


#endif 
