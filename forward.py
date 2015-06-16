import numpy as np


state_num = 3
symbol_num = 2
symbol = np.array([0, 1])

A = np.array([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
B = np.array([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
pi = np.array([1/3, 1/3, 1/3])
obs = np.array([0, 1, 0])

n = len(obs)

c = np.zeros(n)
alpha = np.zeros((n, state_num))

alpha[0, :] = pi[:] * B[:, obs[0]]

for t in range(1, n):
    alpha[t, :] = np.dot(alpha[t-1, :], A) * B[:, obs[t]]

Px = sum(alpha[n-1, :])

print("P(x) = {0:f}".format(Px))
