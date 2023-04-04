import matplotlib.pyplot as plt
from typing import List
from robust_drone_pathfollowing.helpclasses.functions3D import *
import math
from mpl_toolkits.mplot3d import axes3d
import numpy as np


class Wind:
    """Models a specific wind field"""

    ##################################

    def __init__(self, total_force: float, args: int, debug: bool = False):
        """Initialization of a single 3D wind field.

        Parameters
        ----------
        total_force: float
            The maximum amount of force in Newtons that the wind field should have.
        args: int
            The argument that shows which kind of wind the wind field has.
        debug: bool
            Whether the debug messages should be printet.

        """

        self.rand = [random.gauss(total_force/3, 0.03),
                     random.gauss(total_force/3, 0.03),
                     random.gauss(total_force/3, 0.03)]
        self.force = total_force
        self.args = args
        self.sign: List[int] = [random.choice([-1, 1]), random.choice([-1, 1]), random.choice([-1, 1])]
        self.functionX = Function3D(False)
        self.functionY = Function3D(False)
        self.functionZ = Function3D(False)
        if debug:
            self.initPrint()

    def initPrint(self):
        """Prints out the Functions that were used to create the wind field.

        Parameters
        ----------
        self.args: int
            The argument that shows which kind of wind the wind field has.

        """

        if self.args == 0:
            print(self.rand)
        if self.args == 1:
            print(['sin(pi * x) * cos(pi * y) * cos(pi * z)',
                   '-cos(pi * x) * sin(pi * y) * cos(pi * z)',
                   'sqrt(2.0 / 3.0) * cos(pi * x) * cos(pi *  y) * sin(pi * z)'])
        if self.args == 2:
            print(['x * ' + str(self.rand[0]),
                   'y * ' + str(self.rand[1]),
                   'z * ' + str(self.rand[2])])
        if self.args == 3:
            print([str(self.sign[0]) + ' * y',
                   str(self.sign[1]) + ' * x',
                   str(self.sign[2]) + ' * z'])
        if self.args == 4:
            print(['x + ' + str(self.sign[0]) + ' * y',
                   'z + ' + str(self.sign[1]) + ' * x',
                   'y + ' + str(self.sign[2]) + ' * z'])
        if self.args >= 5:
            print([self.functionX.getName(),
                   self.functionY.getName(),
                   self.functionZ.getName()])

    def random(self, x, y, z, plot: bool = False) -> List[float]:
        """A random constant wind field that applies the same random force vector at each position.

        Parameters
        ----------
        x : float / ndarray
            The x position or x position array to calculate the wind vector.
        y : float / ndarray
            The y position or y position array to calculate the wind vector.
        z : float / ndarray
            The z position or z position array to calculate the wind vector.
        plot: bool, optional
            Decides whether the function is used to plot a List of force vectors or just to return a single force vector.

        Returns
        -------
        List[float]: the force vector

        """

        return self.clip(self.rand[0:3])

    def trigo(self, x, y, z, plot: bool = False) -> List:
        """A trigonometric wind field with a central vortex.

        Parameters
        ----------
        x : float / ndarray
            The x position or x position array to calculate the wind vector.
        y : float / ndarray
            The y position or y position array to calculate the wind vector.
        z : float / ndarray
            The z position or z position array to calculate the wind vector.
        plot: bool, optional
            Decides whether the function is used to plot a List of force vectors or just to return a single force vector.

        Returns
        -------
        List[float]: the force vector / List[List[float]]: list of force vectors

        """

        if not plot:
            return self.clip([np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z),
                              - np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z),
                              (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(
                                  np.pi * y) * np.sin(np.pi * z))])
        else:
            return[np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z),
                   - np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z),
                   (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(
                    np.pi * y) * np.sin(np.pi * z))]

    def likelinear(self, x, y, z, plot: bool = False) -> List:
        """A wind field that is linear in each axis.

        Parameters
        ----------
        x : float / ndarray
            The x position or x position array to calculate the wind vector.
        y : float / ndarray
            The y position or y position array to calculate the wind vector.
        z : float / ndarray
            The z position or z position array to calculate the wind vector.
        plot: bool, optional
            Decides whether the function is used to plot a List of force vectors or just to return a single force vector.
        self.rand: List[float]
            Three random numbers that decides how steep each of the linear functions is.

        Returns
        -------
        List[float]: the force vector / List[List[float]]: list of force vectors

        """

        if not plot:
            return self.clip([x * self.rand[0], y * self.rand[1], z * self.rand[2]])
        else:
            return [x * self.rand[0], y * self.rand[1], z * self.rand[2]]

    def vortex(self, x, y, z, plot: bool = False) -> List:
        """A basic - but random wind field with a central vortex.

        Parameters
        ----------
        x : float / ndarray
            The x position or x position array to calculate the wind vector.
        y : float / ndarray
            The y position or y position array to calculate the wind vector.
        z : float / ndarray
            The z position or z position array to calculate the wind vector.
        plot: bool, optional
            Decides whether the function is used to plot a List of force vectors or just to return a single force vector.
        self.sign: List[int]
            Three random signs (-1 or 1) that decides the sign of the function in each axis.

        Returns
        -------
        List[float]: the force vector / List[List[float]]: list of force vectors

        """

        if not plot:
            return self.clip([self.sign[0] * y, self.sign[1] * x, self.sign[2] * z])
        else:
            return [self.sign[0] * y, self.sign[1] * x, self.sign[2] * z]

    def vortex2(self, x, y, z, plot: bool = False) -> List:
        """Another basic - but random wind field with a central vortex.

        Parameters
        ----------
        x : float / ndarray
            The x position or x position array to calculate the wind vector.
        y : float / ndarray
            The y position or y position array to calculate the wind vector.
        z : float / ndarray
            The z position or z position array to calculate the wind vector.
        plot: bool, optional
            Decides whether the function is used to plot a List of force vectors or just to return a single force vector.
        self.sign: List[int]
            Three random signs (-1 or 1) that decides the sign of the function in each axis.

        Returns
        -------
        List[float]: the force vector / List[List[float]]: list of force vectors

        """

        if not plot:
            return self.clip([x + self.sign[0] * y, z + self.sign[1] * x, y + self.sign[2] * z])
        else:
            return [x + self.sign[0] * y, z + self.sign[1] * x, y + self.sign[2] * z]

    def functionwind(self, x, y, z, plot: bool = False):
        """A completely random wind field based on three random 3D functions.

        Parameters
        ----------
        x : float / ndarray
            The x position or x position array to calculate the wind vector.
        y : float / ndarray
            The y position or y position array to calculate the wind vector.
        z : float / ndarray
            The z position or z position array to calculate the wind vector.
        plot: bool, optional
            Decides whether the function is used to plot a List of force vectors or just to return a single force vector.
        self.sign: List[int]
            Three random signs (-1 or 1) that decides the sign of the function in each axis.

        Returns
        -------
        List[float]: the force vector / List[List[float]]: list of force vectors

        """

        if not plot:
            return self.clip(
                [self.sign[0] * self.functionX.apply(x, y, z),
                 self.sign[1] * self.functionY.apply(x, y, z),
                 self.sign[2] * self.functionZ.apply(x, y, z)])
        else:
            return [self.sign[0] * self.functionX.apply(x, y, z),
                    self.sign[1] * self.functionY.apply(x, y, z),
                    self.sign[2] * self.functionZ.apply(x, y, z)]

    def getfunc(self):
        """Returns the math function of the wind field.

        Parameters
        ----------
        self.args: int
            The argument which kind of function is used for the wind field.

        Returns
        -------
        methodfunction: the math function of the wind field.

        """
        if self.args == 0:
            return self.random
        elif self.args == 1:
            return self.trigo
        elif self.args == 2:
            return self.likelinear
        elif self.args == 3:
            return self.vortex
        elif self.args == 4:
            return self.vortex2
        else:
            return self.functionwind

    def clip(self, force_vec: List[float]) -> List[float]:
        """Clips a force vector to the maximum force of the wind field.

        Parameters
        ----------
        force_vec: List[float]
            The force vector that has to be clipped.

        Returns
        -------
        List[float]: the clipped force vector

        """

        length = math.sqrt(force_vec[0]*force_vec[0] + force_vec[1]*force_vec[1] + force_vec[2]*force_vec[2])
        if length > self.force:
            for i, coord in enumerate(force_vec):
                force_vec[i] = random.gauss((coord / length) * self.force, 0.003)
        return force_vec

    def get(self, x: float, y: float, z: float) -> List[float]:
        """Returns the force vector based on the position (x,y,z)

        Parameters
        ----------
        x : float / ndarray
            The x position to calculate the wind vector.
        y : float / ndarray
            The y position to calculate the wind vector.
        z : float / ndarray
            The z position to calculate the wind vector.

        Returns
        -------
        List[float]: the final force vector

        """

        return self.getfunc()(x=x, y=y, z=z, plot=False)

    def plot(self):
        """Plots the not-clipped force field."""

        x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                              np.arange(-0.8, 1, 0.2),
                              np.arange(-0.8, 1, 0.8))
        force_vec = self.getfunc()(x=x, y=y, z=z, plot=True)
        fig = plt.figure()
        #ax = fig.gca(projection='3d')
        ax = fig.add_subplot(projection='3d')
        x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                              np.arange(-0.8, 1, 0.2),
                              np.arange(-0.8, 1, 0.8))
        ax.quiver(x, y, z, force_vec[0], force_vec[1], force_vec[2], length=0.1)
        plt.show()


if __name__ == "__main__":
    for i in range(0, 20):
        wind = Wind(0.5, random.randint(0, 5), True)
        # print(wind.get(0,0,2))
        # print(wind.get(random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)))
        wind.plot()
