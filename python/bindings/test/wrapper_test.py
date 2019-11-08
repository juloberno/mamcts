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

class WrapperTests(unittest.TestCase):
    def import_test(self):
        from mamcts import MctsCrossingStateIntUctUct

    def pickle_unpickle_test(self):
        from mamcts import MctsParameters, CrossingStateParameters
        from mamcts import CrossingStateDefaultParametersFloat, CrossingStateDefaultParametersInt

        params_mcts = MctsParameters()
        params_mcts.DISCOUNT_FACTOR = 0.9
        params_mcts.RANDOM_SEED = 1000
        params_mcts.random_heuristic.MAX_SEARCH_TIME = 10
        params_mcts.random_heuristic.MAX_NUMBER_OF_ITERATIONS = 1000

        params_mcts.uct_statistic.LOWER_BOUND = -1000
        params_mcts.uct_statistic.UPPER_BOUND = 100
        params_mcts.uct_statistic.EXPLORATION_CONSTANT = 0.7

        params_mcts.hypothesis_statistic.COST_BASED_ACTION_SELECTION = False
        params_mcts.hypothesis_statistic.LOWER_COST_BOUND = 0
        params_mcts.hypothesis_statistic.UPPER_COST_BOUND = 1
        params_mcts.hypothesis_statistic.PROGRESSIVE_WIDENING_ALPHA = 0.5
        params_mcts.hypothesis_statistic.PROGRESSIVE_WIDENING_K = 1
        params_mcts.hypothesis_statistic.EXPLORATION_CONSTANT = 0.7

        params_mcts.hypothesis_belief_tracker.RANDOM_SEED_HYPOTHESIS_SAMPLING = 1000
        params_mcts.hypothesis_belief_tracker.HISTORY_LENGTH = 4
        params_mcts.hypothesis_belief_tracker.PROBABILITY_DISCOUNT = 1.0
        params_mcts.hypothesis_belief_tracker.POSTERIOR_TYPE = HypothesisBeliefTracker.PosteriorType.PRODUCT
        params_mcts_unpickle = pu(params_mcts)
        self.assertEqual(params_mcts, params_mcts_unpickle)

        params_crossing_state = CrossingStateDefaultParametersFloat()
        params_crossing_state_unpickle = pu(params_crossing_state)
        self.assertEqual(params_crossing_state, params_crossing_state_unpickle)

        print("test done")

        params_crossing_state = CrossingStateDefaultParametersInt()
        params_crossing_state_unpickle = pu(params_crossing_state)
        self.assertEqual(params_crossing_state, params_crossing_state_unpickle)
        print(params_crossing_state_unpickle)