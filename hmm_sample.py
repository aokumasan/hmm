import numpy as np
import baum_welch_lib as bw

# これが本当の値
# 遷移確率行列
A = np.array([[0.7, 0.3], [0.15, 0.85]])
# 出力確率行列
B = np.array([[0.9, 0.1], [0.3, 0.7]])
# 初期確率
pi = np.array([1/2, 1/2])

# これはパラメータ学習の初期値
# 遷移確率行列
eA = np.array([[0.8, 0.2], [0.1, 0.9]])
# 出力確率行列
eB = np.array([[0.8, 0.2], [0.4, 0.6]])
# 初期確率
epi = np.array([1/2, 1/2])

#np.random.seed(1234)

# create sample
def simulate(nSteps):

    def drawFrom(probs):
        return np.where(np.random.multinomial(1,probs) == 1)[0][0]

    observations = np.zeros(nSteps)
    states = np.zeros(nSteps)
    states[0] = drawFrom(pi)
    observations[0] = drawFrom(B[states[0],:])
    for t in range(1,nSteps):
        states[t] = drawFrom(A[states[t-1],:])
        observations[t] = drawFrom(B[states[t],:])
    return observations,states

# make multiple sequences of observations
o1, s = simulate(1000)
o2, s = simulate(20)
o3, s = simulate(200)
obs = np.array([o1, o2, o3])

hmm = bw.BaumWelch(eA, eB, epi)
hmm.train(obs, 1e-6, 400)

print("Actual parameters")
print(A)
print(B)
print(pi)

print("Estimated parameters")
print(hmm.A)
print(hmm.B)
print(hmm.pi)
