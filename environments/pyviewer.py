# Copyright (c) 2019 Julian Bernhard
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================

import matplotlib.pyplot as plt
from mamcts import Viewer


class PyViewer(Viewer):
    def __init__(self, **kwargs):
        super(PyViewer, self).__init__()
        self.axes = kwargs.pop("axes", plt.subplots(figsize=(20,20))[1])
        self.axes.set_aspect('equal', 'box')

    def drawPoint(self, x,y, size, color):
        self.axes.scatter(x,y, s=50,color="black")
        print("drawing point: {}, {}".format(x,y))
    def drawLine(self, x, y, linewidth, color):
        print("drawing line: {}, {}".format(x,y))
        self.axes.plot(x,y, linewidth=2, color="black")

    def show(self, block=False):
        plt.draw()
        if block:
            plt.show(block=True)
        else:
            plt.pause(0.001)
