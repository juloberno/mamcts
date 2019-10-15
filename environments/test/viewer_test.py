import unittest
from mamcts import CrossingState 
from environments.pyviewer import PyViewer


class PickleTests(unittest.TestCase):
    def test_draw_init(self):
        viewer = PyViewer()
        state = CrossingState({})
        state.draw(viewer)


if __name__ == '__main__':
    unittest.main()