import numpy as np


def dummy_reward(obs, action, i=[0]):
    i[0] += 1

    reward = -np.sum(obs[2])
    if i[0] % 1000 == 0:
        print(reward)
        print(action)
    return reward


def dummy_comfort_reward(obs, action, i=[0]):
    i[0] += 1

    reward = 0

    for j in range(156):
        if (obs[0][j] > 23) & (obs[0][j] < 29):
            reward += 1

    if i[0] % 1000 == 0:
        print(reward)
        print(action)
    return reward
