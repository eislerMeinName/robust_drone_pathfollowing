import matplotlib.pyplot as plt
from typing import List

import numpy
import numpy as np


class PathPlotter:
    """Plotter class that plots the path taken in a single evaluation episode"""

    def __init__(self, goal: List[float]):
        """Initialization of a PathPlotter class.
           Plots path and goal in 3D.

        Parameters
        ----------
           goal: List[float]
               The goal.

        """

        self.fig: plt = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.goal: List[float] = goal
        self.xdata: List[float] = []
        self.ydata: List[float] = []
        self.zdata: List[float] = []

    def addPose(self, pos: List[float]):
        """Method that adds position to the data arrays.

        Parameters
        ----------
        pos: List[float]
            Current Position that should be added to data.

        """

        self.xdata.append(pos[0])
        self.ydata.append(pos[1])
        self.zdata.append(pos[2])

    def getAxScale(self) -> float:
        """Method that calculates the best scaling factor for the axes."""

        return max(np.hstack([self.xdata, self.ydata, self.zdata, self.goal]))

    def show(self):
        """Method that plots the path and the goal."""

        scale: float = self.getAxScale()
        self.ax.scatter3D(self.xdata, self.ydata, self.zdata)
        self.ax.scatter3D(self.goal[0], self.goal[1], self.goal[2], c='red')
        self.ax.plot(self.xdata, self.ydata, self.zdata, color='black')
        self.ax.set_title('drone path and goal')
        self.ax.set_xlim3d(-scale, scale)
        self.ax.set_ylim3d(-scale, scale)
        self.ax.set_zlim3d(0, scale)
        plt.show()


if __name__ == "__main__":
    plotter = PathPlotter([0, 0.5, 0.5])
    plotter.addPose([0, 0, 0])
    plotter.addPose([0, 0, 1])
    plotter.addPose([0, 0, 0.5])
    plotter.show()
