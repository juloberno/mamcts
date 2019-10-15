# Copyright (c) 2019 Julian Bernhard
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
# ========================================================

import matplotlib.pyplot as plt


class CrossingStateViewer:
    def __init__(self, **kwargs):
        self.axes = kwargs.pop("axes", plt.subplots(figsize=(20,20))[1])

    def drawAgent(self, agent_state):

    def drawState(self, crossing_state):
        for agent_state in crossing_state.agents:
            self.drawAgent(agent_state)

        self.drawCrossings(crossing_state)

    def drawCrossings(self, crossing_state):


    def show(self, block=False):
        plt.draw()
        if block:
            plt.show(block=True)
        else:
            plt.pause(0.001)
