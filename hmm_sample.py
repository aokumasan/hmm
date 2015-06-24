import numpy as np
from baum_welch_lib import BaumWelch

# これが本当の値
# 遷移確率行列
A = np.array([[0.7, 0.3], [0.15, 0.85]])
# 出力確率行列
B = np.array([[0.9, 0.1], [0.3, 0.7]])
# 初期確率
pi = np.array([1/2, 1/2])

# これはパラメータ学習の初期値
# 遷移確率行列
eA = np.array([[0.4, 0.6], [0.3, 0.7]])
# 出力確率行列
eB = np.array([[0.8, 0.2], [0.2, 0.8]])
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
o1, s = simulate(50)
o2, s = simulate(20)
o3, s = simulate(9)
o4, s = simulate(3)
obs = np.array([o1, o2, o3, o4])

hmm = BaumWelch(eA, eB, epi)
hmm.train(obs, 1e-4, 400)

print("Actual parameter [pi]")
print(pi)
print("Estimated parameter [pi]")
print(hmm.pi)
print("")
print("Actual parameter [A]")
print(A)
print("Estimated parameter [A]")
print(hmm.A)
print("")
print("Actual parameter [B]")
print(B)
print("Estimated parameter [B]")
print(hmm.B)

