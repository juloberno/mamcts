# Copyright (c) 2019 Julian Bernhard
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================

import matplotlib.pyplot as plt


class Viewer(PyViewer):
    def __init__(self, **kwargs):
        self.axes = kwargs.pop("axes", plt.subplots(figsize=(20,20))[1])

    def drawPoint(self, x,y, size, color, alpha):
        pass
    def drawLine(self, x, y, linewidth, color, alpha):
        pass

    def show(self, block=False):
        plt.draw()
        if block:
            plt.show(block=True)
        else:
            plt.pause(0.001)
