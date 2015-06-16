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

# scaled forwardアルゴリズムでスケーリング係数cを求める
# 変数の初期化
alpha = np.zeros((n, state_num))
c = np.zeros(n)

# 初期化
alpha[0, :] = pi[:] * B[:, obs[0]]
c[0] = 1.0 / np.sum(alpha[0, :])
alpha[0, :] = c[0] * alpha[0, :]

# 再帰的計算
for t in range(1, n):
    alpha[t, :] = np.dot(alpha[t-1, :], A) * B[:, obs[t]]
    c[t] = 1.0 / np.sum(alpha[t, :])
    alpha[t, :] = c[t] * alpha[t, :]


# ここからscaled backwardアルゴリズム    
# 変数の初期化
beta = np.zeros((n, state_num))

# 初期化
beta[n-1, :] = c[n-1]

# 再帰的計算
for t in range((n-1), 0, -1):
    beta[t-1, :] = np.dot(A, (B[:, obs[t]] * beta[t, :]))
    beta[t-1, :] = c[t-1] * beta[t-1, :]

# 確率の計算
Px = 1 / np.prod(c[:])

print("P(x) = {0:f}".format(Px))
