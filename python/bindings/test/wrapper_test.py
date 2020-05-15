# Copyright (c) 2019 Julian Bernhard
# 
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================

import unittest
import pickle

def pu(object):
    with open("tmp.pickle","wb") as file:
        pickle.dump(object, file)
    object_unpickle = None
    with open("tmp.pickle", "rb") as file:
        object_unpickle = pickle.load(file)
    return object_unpickle

def is_equal_mcts_params(mctsp1, mctsp2):
    return mctsp1.DISCOUNT_FACTOR == mctsp2.DISCOUNT_FACTOR and \
        mctsp1.RANDOM_SEED == mctsp2.RANDOM_SEED and \
        mctsp1.MAX_SEARCH_TIME == mctsp2.MAX_SEARCH_TIME and \
        mctsp1.MAX_NUMBER_OF_ITERATIONS == mctsp2.MAX_NUMBER_OF_ITERATIONS and \
        mctsp1.MAX_SEARCH_DEPTH == mctsp2.MAX_SEARCH_DEPTH and \
        mctsp1.random_heuristic.MAX_SEARCH_TIME == mctsp2.random_heuristic.MAX_SEARCH_TIME and \
        mctsp1.random_heuristic.MAX_NUMBER_OF_ITERATIONS == mctsp2.random_heuristic.MAX_NUMBER_OF_ITERATIONS and \
        mctsp1.uct_statistic.LOWER_BOUND == mctsp2.uct_statistic.LOWER_BOUND and \
        mctsp1.uct_statistic.UPPER_BOUND == mctsp2.uct_statistic.UPPER_BOUND and \
        mctsp1.uct_statistic.EXPLORATION_CONSTANT == mctsp2.uct_statistic.EXPLORATION_CONSTANT and \
        mctsp1.hypothesis_statistic.COST_BASED_ACTION_SELECTION == mctsp2.hypothesis_statistic.COST_BASED_ACTION_SELECTION and \
        mctsp1.hypothesis_statistic.PROGRESSIVE_WIDENING_HYPOTHESIS_BASED == mctsp2.hypothesis_statistic.PROGRESSIVE_WIDENING_HYPOTHESIS_BASED and \
        mctsp1.hypothesis_statistic.LOWER_COST_BOUND == mctsp2.hypothesis_statistic.LOWER_COST_BOUND and \
        mctsp1.hypothesis_statistic.UPPER_COST_BOUND == mctsp2.hypothesis_statistic.UPPER_COST_BOUND and \
        mctsp1.hypothesis_statistic.PROGRESSIVE_WIDENING_ALPHA == mctsp2.hypothesis_statistic.PROGRESSIVE_WIDENING_ALPHA and \
        mctsp1.hypothesis_statistic.PROGRESSIVE_WIDENING_K == mctsp2.hypothesis_statistic.PROGRESSIVE_WIDENING_K and \
        mctsp1.hypothesis_statistic.EXPLORATION_CONSTANT == mctsp2.hypothesis_statistic.EXPLORATION_CONSTANT and \
        mctsp1.hypothesis_belief_tracker.RANDOM_SEED_HYPOTHESIS_SAMPLING == \
                 mctsp2.hypothesis_belief_tracker.RANDOM_SEED_HYPOTHESIS_SAMPLING and \
        mctsp1.hypothesis_belief_tracker.HISTORY_LENGTH == mctsp2.hypothesis_belief_tracker.HISTORY_LENGTH and \
        mctsp1.hypothesis_belief_tracker.PROBABILITY_DISCOUNT == mctsp2.hypothesis_belief_tracker.PROBABILITY_DISCOUNT and \
        mctsp1.hypothesis_belief_tracker.POSTERIOR_TYPE == mctsp2.hypothesis_belief_tracker.POSTERIOR_TYPE and \
        mctsp1.hypothesis_belief_tracker.FIXED_HYPOTHESIS_SET == mctsp2.hypothesis_belief_tracker.FIXED_HYPOTHESIS_SET

def is_equal_crossing_state_params(cp1, cp2):
    return cp1.NUM_OTHER_AGENTS == cp2.NUM_OTHER_AGENTS and \
        cp1.OTHER_AGENTS_POLICY_RANDOM_SEED == cp2.OTHER_AGENTS_POLICY_RANDOM_SEED and \
        cp1.COST_ONLY_COLLISION == cp2.COST_ONLY_COLLISION  and \
        cp1.MAX_VELOCITY_EGO == cp2.MAX_VELOCITY_EGO  and \
        cp1.MIN_VELOCITY_EGO == cp2.MIN_VELOCITY_EGO  and \
        cp1.MIN_VELOCITY_OTHER == cp2.MIN_VELOCITY_OTHER  and \
        cp1.MAX_VELOCITY_OTHER == cp2.MAX_VELOCITY_OTHER  and \
        cp1.EGO_GOAL_POS == cp2.EGO_GOAL_POS and \
        cp1.CHAIN_LENGTH == cp2.CHAIN_LENGTH   and \
        cp1.NUM_OTHER_ACTIONS ==  cp2.NUM_OTHER_ACTIONS and \
        cp1.REWARD_COLLISION ==  cp2.REWARD_COLLISION and \
        cp1.REWARD_GOAL_REACHED ==  cp2.REWARD_GOAL_REACHED and \
        cp1.REWARD_STEP ==  cp2.REWARD_STEP

class WrapperTests(unittest.TestCase):
    def test_import(self):
        from mamcts import MctsCrossingStateIntUctUct

    def test_pickle_unpickle(self):
        from mamcts import MctsParameters, HypothesisBeliefTracker
        from mamcts import CrossingStateDefaultParametersFloat, CrossingStateDefaultParametersInt

        params_mcts = MctsParameters()
        params_mcts.DISCOUNT_FACTOR = 0.9
        params_mcts.RANDOM_SEED = 1000
        params_mcts.MAX_SEARCH_TIME = 1232423
        params_mcts.MAX_NUMBER_OF_ITERATIONS = 2315677
        params_mcts.random_heuristic.MAX_SEARCH_TIME = 10
        params_mcts.random_heuristic.MAX_NUMBER_OF_ITERATIONS = 1000

        params_mcts.uct_statistic.LOWER_BOUND = -1000
        params_mcts.uct_statistic.UPPER_BOUND = 100
        params_mcts.uct_statistic.EXPLORATION_CONSTANT = 0.7

        params_mcts.hypothesis_statistic.COST_BASED_ACTION_SELECTION = True
        params_mcts.hypothesis_statistic.PROGRESSIVE_WIDENING_HYPOTHESIS_BASED = True
        params_mcts.hypothesis_statistic.LOWER_COST_BOUND = 0
        params_mcts.hypothesis_statistic.UPPER_COST_BOUND = 1
        params_mcts.hypothesis_statistic.PROGRESSIVE_WIDENING_ALPHA = 0.5
        params_mcts.hypothesis_statistic.PROGRESSIVE_WIDENING_K = 1
        params_mcts.hypothesis_statistic.EXPLORATION_CONSTANT = 0.7

        params_mcts.hypothesis_belief_tracker.RANDOM_SEED_HYPOTHESIS_SAMPLING = 1000
        params_mcts.hypothesis_belief_tracker.HISTORY_LENGTH = 4
        params_mcts.hypothesis_belief_tracker.PROBABILITY_DISCOUNT = 1.0
        params_mcts.hypothesis_belief_tracker.POSTERIOR_TYPE = HypothesisBeliefTracker.PosteriorType.PRODUCT
        params_mcts.hypothesis_belief_tracker.FIXED_HYPOTHESIS_SET = {1: 5, 10: 4, 3 : 100}
        params_mcts_unpickle = pu(params_mcts)
        self.assertTrue(is_equal_mcts_params(params_mcts, params_mcts_unpickle))

        params_crossing_state = CrossingStateDefaultParametersFloat()
        params_crossing_state_unpickle = pu(params_crossing_state)
        self.assertTrue(is_equal_crossing_state_params(params_crossing_state, params_crossing_state_unpickle))

        params_crossing_state = CrossingStateDefaultParametersInt()
        params_crossing_state_unpickle = pu(params_crossing_state)
        self.assertTrue(is_equal_crossing_state_params(params_crossing_state, params_crossing_state_unpickle))

if __name__ == '__main__':
    unittest.main()