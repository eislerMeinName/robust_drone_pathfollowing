import numpy as np
import matplotlib.pyplot as plt
Goal = np.array([0, 0, 0.5])
from mpl_toolkits.mplot3d import Axes3D


def rewardfunction(x: np.array, y: np.array, z: np.array) -> np.array:
    reward: np.array = np.array([])
    for i, element in enumerate(x):
        cur_reward: float = 0
        if z[i] <= 0.18:
            cur_reward += -100
        cur_reward += -1 * np.linalg.norm(Goal - np.array([x[i], y[i], z[i]])) ** 2
        reward = np.append(reward, np.array([cur_reward]))
    return reward


if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    x = np.array([])
    y = np.array([])
    z = np.array([])
    for ix in range(0, 20):
        for iy in range(0, 20):
            for iz in range(0, 20):
                x = np.append(x, np.array([-1 + 0.1 * ix]))
                y = np.append(y, np.array([-1 + 0.1 * iy]))
                z = np.append(z, np.array([0 + 0.05 * iz]))

    c = rewardfunction(x, y, z)

    img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()

