import random
import numpy
import numpy as np


class Function3D :
    """Models a random 3D function"""

    def __init__(self, is_inner: bool) -> None:
        """Initialization of a single top 3D function.

        Parameters
        ----------
        is_inner: bool
            to decide whether the 3D function is a top or inner function and is used to clip the recursion.

        """

        # outer function with a max of 2 inner function and a random argument that decides which function it is
        if not is_inner:
            self.inner = Function3D(True)
            self.inner2 = Function3D(True)
            self.arg = random.randint(0, 15)
        # inner function with a random argument that decides between chosen inner functions
        else:
            self.arg = random.randint(0, 9)

    def zero(self, x, y, z):
        """Method that implements a zero function.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.

            Returns
            -------
                0 / ndarray of 0

        """
        return x * 0

    def one(self, x , y, z):
        """Method that implements a one function.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.

            Returns
            -------
                1 / ndarray of 1

        """

        if type(x) != float and type(x) != int:
            np.ones(x.size)
        return 1

    def sum(self, x, y, z):
        """Method that implements a simple sum function that adds all the input parameters.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.

            Returns
            -------
                x+y+z / ndarray of (x+y+z)

        """

        return x+y+z

    def sumXY(self, x, y, z):
        """Method that implements a simple sum function that adds the input parameters x and y.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.

            Returns
            -------
                x+y / ndarray of (x+y)

        """

        return x+y

    def sumXZ(self, x, y, z):
        """Method that implements a simple sum function that adds the input parameters x and z.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.

            Returns
            -------
                x+z / ndarray of (x+z)

        """

        return x+z

    def sumYZ(self, x, y, z):
        """Method that implements a simple sum function that adds the input parameters y and z.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.

            Returns
            -------
                y+z / ndarray of (y+z)

        """

        return y+z

    def mult(self, x, y, z):
        """Method that implements a simple multiplication function that multiplies all the input parameters.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.

            Returns
            -------
                x*y*z / ndarray of (x*y*z)

        """

        return x*y*z

    def multXY(self, x, y, z):
        """Method that implements a simple multiplication function that multiplies the input parameters x and y.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.

            Returns
            -------
                x*y / ndarray of (x*y)

        """

        return x*y

    def multXZ(self, x, y, z):
        """Method that implements a simple multiplication function that multiplies the input parameters x and z.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.

            Returns
            -------
                x*z / ndarray of (x*z)

        """

        return x*z

    def multYZ(self, x, y, z):
        """Method that implements a simple multiplication function that multiplies the input parameters y and z.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.

            Returns
            -------
                y*z / ndarray of (y*z)

        """

        return y*z

    def sin(self, x, y, z):
        """Method that implements a simple sin function.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.
            self.inner: function3D
                A random simple 3D function.

            Returns
            -------
                sin(innerfunction(x,y,z)) / ndarray of sin(innerfunction(x,y,z))

        """

        return np.sin(self.inner.apply(x, y, z))

    def cos(self, x, y, z):
        """Method that implements a simple cos function.

             Parameters
             ----------
             x : float / ndarray
                 The x position or x position array to calculate the function value.
             y : float / ndarray
                 The y position or y position array to calculate the function value.
             z : float / ndarray
                 The z position or z position array to calculate the function value.
             self.inner: function3D
                 A random simple 3D function.

             Returns
             -------
                 cos(innerfunction(x,y,z)) / ndarray of cos(innerfunction(x,y,z))

        """

        return np.cos(self.inner.apply(x, y, z))

    def specialSin(self, x, y, z):
        """Method that implements a special sin function.

             Parameters
             ----------
             x : float / ndarray
                 The x position or x position array to calculate the function value.
             y : float / ndarray
                 The y position or y position array to calculate the function value.
             z : float / ndarray
                 The z position or z position array to calculate the function value.
             self.inner: function3D
                 A random simple 3D function.

             Returns
             -------
                 2*x*sin(innerfunction(x,y,z)) / ndarray of 2*x*sin(innerfunction(x,y,z))

        """

        return (2*x) * self.sin(x, y, z)

    def add2(self, x, y, z):
        """Method that implements simple add function that multiplies the two inner functions.

             Parameters
             ----------
             x : float / ndarray
                 The x position or x position array to calculate the function value.
             y : float / ndarray
                 The y position or y position array to calculate the function value.
             z : float / ndarray
                 The z position or z position array to calculate the function value.
             self.inner: function3D
                 A random simple 3D function.
             self.inner2: function3D
                 A random simple 3D function.

             Returns
             -------
                 innerfunction(x,y,z) + innerfunction2(x,y,z) / ndarray of innerfunction(x,y,z) + innerfunction2(x,y,z)

        """

        return self.inner.apply(x, y, z) + self.inner2.apply(x, y, z)

    def sqrtnum(self, x, y, z):
        """Method that implements a simple sqrt function that uses an absolute value.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.
            self.inner: function3D
                A random simple 3D function.

            Returns
            -------
                sqrt(innerfunction(x,y,z)) or sqrt(-innerfunction(x,y,z))

        """

        if self.inner.apply(x, y, z) < 0:
            return np.sqrt(-self.inner.apply(x, y, z))
        else:
            return np.sqrt(self.inner.apply(x, y, z))

    def sqrtplot(self, x, y, z):
        """Method that implements a simple sqrt function that uses an absolute value for plotting.

            Parameters
            ----------
            x : ndarray
                The x position or x position array to calculate the function value.
            y : ndarray
                The y position or y position array to calculate the function value.
            z : ndarray
                The z position or z position array to calculate the function value.
            self.inner: function3D
                A random simple 3D function.

            Returns
            -------
                ndarray of sqrt(|innerfunction(x,y,z)|)

        """

        return np.sqrt(np.absolute(self.inner.apply(x, y, z)))

    def exp(self, x, y, z):
        """Method that implements a simple exp function that uses an absolute value.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.
            self.inner: function3D
                A random simple 3D function.

            Returns
            -------
                e^innerfunction(x,y,z) / ndarray of e^innerfunction(x,y,z)

        """

        return np.exp(self.inner.apply(x, y, z))

    def apply(self, x, y, z):
        """An apply method that applies the function based on the input parameters and the functions argument.

            Parameters
            ----------
            x : float / ndarray
                The x position or x position array to calculate the function value.
            y : float / ndarray
                The y position or y position array to calculate the function value.
            z : float / ndarray
                The z position or z position array to calculate the function value.
            self.inner: function3D
                A random simple 3D function.
            self.inner2: function3D
                A random simple 3D function.
            self.arg: int
                The functions argument that shows which function it is.

            Returns
            -------
                float: the value / ndarray: array of values

        """

        if self.arg == 0:
            return self.zero(x, y, z)
        if self.arg == 1:
            return self.one(x, y, z)
        if self.arg == 2:
            return self.sum(x, y, z)
        if self.arg == 3:
            return self.sumXY(x=x, y=y, z=z)
        if self.arg == 4:
            return self.sumXZ(x=x, z=z, y=y)
        if self.arg == 5:
            return self.sumYZ(y=y, z=z, x=x)
        if self.arg == 6:
            return self.mult(x, y, z)
        if self.arg == 7:
            return self.multXY(x=x, y=y, z=z)
        if self.arg == 8:
            return self.multXZ(x=x, z=z, y=y)
        if self.arg == 9:
            return self.multYZ(y=y, z=z, x=x)
        if self.arg == 10:
            return self.sin(x, y, z)
        if self.arg == 11:
            return self.cos(x, y, z)
        if self.arg == 12:
            return self.specialSin(x, y, z)
        if self.arg == 13:
            return self.add2(x, y, z)
        if self.arg == 14:
            if type(x) == numpy.ndarray:
                return self.sqrtplot(x, y, z)
            else:
                return self.sqrtnum(x, y, z)
        if self.arg == 15:
            return self.exp(x, y, z)

    def getName(self) -> str:
        """A method that provides the name of the 3Dfunction.

            Parameters
            ----------
            self.inner: function3D
                A random simple 3D function.
            self.inner2: function3D
                A random simple 3D function.
            self.arg: int
                The functions argument that shows which function it is.

            Returns
            -------
                str: A string that shows the current 3Dfunctions name.

        """

        if self.arg == 0:
            return "0"
        if self.arg == 1:
            return "1"
        if self.arg == 2:
            return "x + y + z"
        if self.arg == 3:
            return "x + y"
        if self.arg == 4:
            return "x + z"
        if self.arg == 5:
            return "y + z"
        if self.arg == 6:
            return "x * y * z"
        if self.arg == 7:
            return "x * y"
        if self.arg == 8:
            return "x * z"
        if self.arg == 9:
            return "y * z"
        if self.arg == 10:
            return "sin(" + self.inner.getName() + ")"
        if self.arg == 11:
            return "cos(" + self.inner.getName() + ")"
        if self.arg == 12:
            return "(2*x) * sin(" + self.inner.getName() + ")"
        if self.arg == 13:
            return "(" + self.inner.getName() + ") * (" + self.inner2.getName() + ")"
        if self.arg == 14:
            return "sqrt(" + self.inner.getName() + ")"
        if self.arg == 15:
            return "exp(" + self.inner.getName() + ")"
