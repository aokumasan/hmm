import numpy as np

# 状態
omega = np.array([0, 1, 2])
# 状態数
state_num = 3
# 出力シンボル(この場合は 0 と 1 )
symbol = np.array([0, 1])
# 出力シンボルの数
symbol_num = 2

# 遷移確率行列
A = np.matrix([[0.1, 0.7, 0.2], [0.2, 0.1, 0.7], [0.7, 0.2, 0.1]])
# 出力確率行列
B = np.matrix([[0.9, 0.1], [0.6, 0.4], [0.1, 0.9]])
# 初期確率
pi = np.matrix([1/3, 1/3, 1/3])
# 観測系列
obs = np.array([0, 1, 0])
# 観測系列の長さ
n = len(obs)



# 初期化
psi = np.zeros((n, state_num))
c_psi = np.zeros((n, state_num))
psi[0, :] = np.multiply(pi[:], B[:, obs[0]].T)


# 再帰的計算
for t in range(1, n):
    for j in range(0, state_num):
        psi[t, j] = np.max(np.multiply(psi[t-1, :], A[:, j].T)) * B[j, obs[t]]
        c_psi[t, j] = np.argmax(np.multiply(psi[t-1, :], A[:, j].T))


# 終了
print("P(x,s*) = ", np.max(psi[n-1, :]))
i = np.zeros(state_num)
s = np.zeros(state_num)
i[n-1] = np.argmax(psi[n-1, :])
s[n-1] = omega[i[n-1]]


# 系列の復元
for t in range(n-1, 0, -1):
    i[t-1] = c_psi[t, i[t]]
    s[t-1] = omega[i[t-1]]
