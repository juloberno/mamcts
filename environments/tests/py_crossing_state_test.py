import unittest
from mamcts import CrossingState 
from environments.pyviewer import PyViewer


class PickleTests(unittest.TestCase):
    def test_draw_state(self):
        viewer = PyViewer()
        state = CrossingState({})
        state.draw(viewer)
        viewer.show(block=True)

    def test_episode_runner(self):
        viewer = PyViewer()
        runner = CrossingStateEpisodeRunner(
            {1 : AgentPolicyCrossingState(5,5),
             2 : AgentPolicyCrossingState(5,5) },
            [AgentPolicyCrossingState(4,5), 
             AgentPolicyCrossingState(5,6)]
             30,
             viewer)
        for _ in range(0,10):
            runner.step()
            viewer.show()


if __name__ == '__main__':
    unittest.main()