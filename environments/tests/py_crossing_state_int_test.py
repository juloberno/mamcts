import unittest
from mamcts import CrossingStateInt, CrossingStateEpisodeRunnerInt
from mamcts import AgentPolicyCrossingStateInt, CrossingStateParametersInt
from mamcts import CrossingStateParametersInt
from mamcts import HypothesisBeliefTracker
from environments.pyviewer import PyViewer


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
             None)

        episode_result = runner.run()
        print(runner.EVAL_RESULT_COLUMN_DESC)
        print(episode_result)

if __name__ == '__main__':
    unittest.main()