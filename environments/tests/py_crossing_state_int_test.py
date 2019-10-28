import unittest
from mamcts import CrossingStateInt, CrossingStateEpisodeRunnerInt
from mamcts import AgentPolicyCrossingStateInt, CrossingStateParametersInt
from mamcts import CrossingStateParametersInt
from mamcts import HypothesisBeliefTracker
from environments.pyviewer import PyViewer
from mamcts import MctsParameters

def default_mcts_parameters():
    parameters = MctsParameters()
    parameters.DISCOUNT_FACTOR = 0.9
    parameters.random_heuristic.MAX_SEARCH_TIME = 10
    parameters.random_heuristic.MAX_NUMBER_OF_ITERATIONS = 1000

    parameters.uct_statistic.LOWER_BOUND = -1000
    parameters.uct_statistic.UPPER_BOUND = 100
    parameters.uct_statistic.EXPLORATION_CONSTANT = 0.7

    parameters.hypothesis_statistic.COST_BASED_ACTION_SELECTION = False
    parameters.hypothesis_statistic.LOWER_COST_BOUND = 0
    parameters.hypothesis_statistic.UPPER_COST_BOUND = 1
    parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_ALPHA = 0.5
    parameters.hypothesis_statistic.PROGRESSIVE_WIDENING_K = 1
    parameters.hypothesis_statistic.EXPLORATION_CONSTANT = 0.7

    return parameters
class PickleTests(unittest.TestCase):
    def test_draw_state(self):
        viewer = PyViewer()
        state = CrossingStateInt({})
        state.draw(viewer)
        viewer.show(block=True)

    def test_episode_runner_step(self):
        CrossingStateParametersInt.CHAIN_LENGTH = 21
        viewer = PyViewer()
        runner = CrossingStateEpisodeRunnerInt(
            {1 : AgentPolicyCrossingStateInt((5,5)),
             2 : AgentPolicyCrossingStateInt((5,5)) },
            [AgentPolicyCrossingStateInt((4,5)), 
             AgentPolicyCrossingStateInt((5,6))],
             30,
             4,
             1.0,
             HypothesisBeliefTracker.PosteriorType.PRODUCT,
             10000,
             10000,
             default_mcts_parameters(),
             viewer)
        for _ in range(0, 20):
            viewer.clear()
            runner.step()
            viewer.show()

    def test_episode_runner_run(self):
        CrossingStateParametersInt.CHAIN_LENGTH = 21
        runner = CrossingStateEpisodeRunnerInt(
            {1 : AgentPolicyCrossingStateInt((5,5)),
             2 : AgentPolicyCrossingStateInt((5,5)) },
            [AgentPolicyCrossingStateInt((4,5)), 
             AgentPolicyCrossingStateInt((5,6))],
             30,
             4,
             1.0,
             HypothesisBeliefTracker.PosteriorType.PRODUCT,
             10000,
             10000,
             default_mcts_parameters(),
             None)

        episode_result = runner.run()
        print(runner.EVAL_RESULT_COLUMN_DESC)
        print(episode_result)

if __name__ == '__main__':
    unittest.main()