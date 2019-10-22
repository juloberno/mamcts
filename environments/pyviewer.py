# Copyright (c) 2019 Julian Bernhard
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

from mamcts import Viewer


class PyViewer(Viewer):
    def __init__(self, **kwargs):
        super(PyViewer, self).__init__()
        self.axes = kwargs.pop("axes", plt.subplots(figsize=(20,20))[1])
        self.axes_x_limits = kwargs.pop("xlim", [-12, 12])
        self.axes_y_limits = kwargs.pop("ylim", [-12, 12] )
        self.axes.set_aspect('equal', 'box')

    def drawPoint(self, x,y, size, color):
        self.axes.scatter(x,y, s=size,color=(color[0],color[1], color[2]))
    def drawLine(self, x, y, linewidth, color):
        self.axes.plot(x,y, linewidth=linewidth, color=(color[0],color[1], color[2]))

    def clear(self):
        self.axes.cla()

    def show(self, block=False):
        plt.draw()
        self.axes.set_xlim(self.axes_x_limits)
        self.axes.set_ylim(self.axes_y_limits)
        if block:
            plt.show(block=True)
        else:
            plt.pause(0.001)
