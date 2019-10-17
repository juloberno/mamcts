import unittest
from mamcts import CrossingState, CrossingStateEpisodeRunner, AgentPolicyCrossingState, CrossingStateParameters, HypothesisBeliefTracker
from environments.pyviewer import PyViewer


class PickleTests(unittest.TestCase):
    @unittest.skip
    def test_draw_state(self):
        viewer = PyViewer()
        state = CrossingState({})
        state.draw(viewer)
        viewer.show(block=True)
    @unittest.skip
    def test_episode_runner_step(self):
        CrossingStateParameters.CHAIN_LENGTH = 21
        viewer = PyViewer()
        runner = CrossingStateEpisodeRunner(
            {1 : AgentPolicyCrossingState((5,5)),
             2 : AgentPolicyCrossingState((5,5)) },
            [AgentPolicyCrossingState((4,5)), 
             AgentPolicyCrossingState((5,6))],
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
        CrossingStateParameters.CHAIN_LENGTH = 21
        runner = CrossingStateEpisodeRunner(
            {1 : AgentPolicyCrossingState((5,5)),
             2 : AgentPolicyCrossingState((5,5)) },
            [AgentPolicyCrossingState((4,5)), 
             AgentPolicyCrossingState((5,6))],
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