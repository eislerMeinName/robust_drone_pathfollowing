from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def reward(x: np.array, y: np.array, z: np.array) -> np.array:
    ret: List = []
    goal: np.array = np.array([0, 0, 0.5])

    dist: float = 20
    for i, el in enumerate(x):
        state: np.array = np.array([x[i], y[i], z[i]])
        ret.append(1 - np.exp2(np.linalg.norm(goal - state) / dist))

    return np.array(ret)


def rewXY(x: np.array, y: np.array) -> np.array:
    ret: List = []
    dist: float = 20
    return (1 - np.exp2(np.sqrt(x**2 + y**2) / dist))


def rewXZ(x: np.array, y: np.array) -> np.array:
    dist: float = 20
    return 1 - np.exp2(np.sqrt(x**2 + (0.5 - y)**2) / dist)


def rewdist(dists: np.array) -> np.array:
    #dist: float = 20
    #return 1 - np.exp2(dists/ dist)
    return np.exp(-0.8 * dists) - 1


def dplot():
    x: np.array = np.arange(-1, 1, 0.1)
    y: np.array = np.arange(-1, 1, 0.1)
    z: np.array = np.arange(0, 1, 0.05)
    x, y, z = np.meshgrid(x, y, z)
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()
    rew: np.array = reward(x, y, z)

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")

    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.3,
            alpha=0.2)

    sctt = ax.scatter3D(x, y, z,
                        alpha=0.8,
                        c=rew,
                        cmap="plasma"
                        )
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('z[m]')
    fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)

    # show plot
    plt.show()


def xyplot():

    x: np.array = np.arange(-1, 1, 0.05)
    y: np.array = np.arange(-1, 1, 0.05)
    x, y = np.meshgrid(x, y)
    #x = x.flatten()
    #y = y.flatten()
    rew = rewXY(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, rew, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')
    ax.set_zlabel('reward')
    # rotation
    ax.view_init(30, 60)
    plt.savefig('3Dplot1.png', dpi=600)
    plt.show()


def xzplot():
    x: np.array = np.arange(-1, 1, 0.05)
    z: np.array = np.arange(0, 1, 0.025)
    x, y = np.meshgrid(x, z)
    # x = x.flatten()
    # y = y.flatten()
    rew = rewXZ(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, rew, cmap="plasma", linewidth=0, antialiased=False, alpha=0.5)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('z[m]')
    ax.set_zlabel('reward')
    # rotation
    ax.view_init(30, 60)
    plt.show()


def distplot():
    dist: np.array = np.arange(0, 2, 0.1)
    rew: np.array = rewdist(dist)
    plt.plot(dist, rew)
    plt.xlabel('distance[m]')
    plt.ylabel('reward')
    plt.show()


if __name__ == "__main__":
    dplot()
    xyplot()
    xzplot()
    distplot()

