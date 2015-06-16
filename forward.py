import numpy as np

# 状態数
state_num = 3
# 出力シンボル(この場合は 0 と 1 )
symbol = np.array([0, 1])
# 出力シンボルの数
symbol_num = 2

# 遷移確率行列
A = np.array([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
# 出力確率行列
B = np.array([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
# 初期確率
pi = np.array([1/3, 1/3, 1/3])
# 観測系列
obs = np.array([0, 1, 0])
# 観測系列の長さ
n = len(obs)

# 変数の初期化
c = np.zeros(n)
alpha = np.zeros((n, state_num))

# 初期化
alpha[0, :] = pi[:] * B[:, obs[0]]

# 再帰的計算
for t in range(1, n):
    alpha[t, :] = np.dot(alpha[t-1, :], A) * B[:, obs[t]]

# 確率の計算
Px = sum(alpha[n-1, :])

print("P(x) = {0:f}".format(Px))
