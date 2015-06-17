import numpy as np
import sys

# 状態数
state_num = 2
# 出力シンボル(この場合は 0 と 1 )
symbol = np.array([0, 1])
# 出力シンボルの数
symbol_num = len(symbol)

# これが本当の値
# 遷移確率行列
A = np.array([[0.85, 0.15], [0.12, 0.88]])
# 出力確率行列
B = np.array([[0.8, 0.2], [0.4, 0.6]])
# 初期確率
pi = np.array([1/2, 1/2])

# これはパラメータ学習の初期値
# 遷移確率行列
eA = np.array([[0.7, 0.3], [0.4, 0.6]])
# 出力確率行列
eB = np.array([[0.6, 0.4], [0.5, 0.5]])
# 初期確率
epi = np.array([1/2, 1/2])


np.random.seed(1234)

class HMM:
    # constructor
    def __init__(self, A, B, pi):
        self.A = A     # transition matrix
        self.B = B     # emission matrix
        self.pi = pi   # start prob


    # create HMM sample (obervations & states)
    def sample(self, steps):

        def drawFrom(probs):
            return np.where(np.random.multinomial(1,probs) == 1)[0][0]

        observations = np.zeros(steps)
        states = np.zeros(steps)
        states[0] = drawFrom(self.pi)
        observations[0] = drawFrom(self.B[states[0],:])
        for t in range(1,steps):
            states[t] = drawFrom(self.A[states[t-1],:])
            observations[t] = drawFrom(self.B[states[t],:])

        return observations, states


    # scaled forward algorithm
    def forward(self, obs):
        # length of observations
        n = len(obs)

        # init variables
        alpha = np.zeros((n, state_num))
        c = np.zeros(n)

        # initialization
        alpha[0, :] = self.pi[:] * self.B[:, obs[0]]
        c[0] = 1.0 / np.sum(alpha[0, :])
        alpha[0, :] = c[0] * alpha[0, :]

        # induction
        for t in range(1, n):
            alpha[t, :] = np.dot(alpha[t-1, :], self.A) * self.B[:, obs[t]]
            c[t] = 1.0 / np.sum(alpha[t, :])
            alpha[t, :] = c[t] * alpha[t, :]

        self.alpha = alpha
        self.c = c


    # scaled backward algorithm
    def backward(self, obs):
        # length of observations
        n = len(obs)

        # init variables
        beta = np.zeros((n, state_num))

        # initialization
        try:
            beta[n-1, :] = self.c[n-1]
        except AttributeError:
            print("Error: scaling value is undefined. Please use this function after use forward().")
            sys.exit()

        # induction
        for t in range((n-1), 0, -1):
            beta[t-1, :] = np.dot(self.A, (self.B[:, obs[t]] * beta[t, :]))
            beta[t-1, :] = self.c[t-1] * beta[t-1, :]

        self.beta = beta


    # M-Step
    def __baum_welch_mstep(self, obs):
        # length of observations
        n = len(obs)

        # update A
        newA = numer = denom = np.zeros((state_num, state_num))
        for t in range(0, n-1):
            numer = numer + self.alpha[t, :][:, np.newaxis] * self.A * self.B[:, obs[t+1]] * self.beta[t+1, :]
            denom = denom + self.alpha[t, :] * self.beta[t, :] / self.c[t]
        newA = numer / denom.T

        # update B
        newB = np.zeros((state_num, symbol_num))
        for j in range(0, state_num):
            for k in range(0, symbol_num):
                numer2 = denom2 = 0
                for t in range(0, n):
                    if obs[t] == symbol[k]:
                        numer2 = numer2 + self.alpha[t, j] * self.beta[t, j] / self.c[t]
                    denom2 = denom2 + self.alpha[t, j].T * self.beta[t, j] / self.c[t]
                newB[j, k] = numer2 / denom2

        # update pi
        newpi = self.alpha[0, :] * self.beta[0, :] / self.c[0]

        self.A = newA
        self.B = newB
        self.pi = newpi


    # Baum Welch algorithm
    def baum_welch(self, obs, eps = 1e-9, max_iter = 400):

        old_loglikelihood = 0.0
        for i in range(0, max_iter):
            # E-Step
            self.forward(obs)
            self.backward(obs)
            # M-Step
            self.__baum_welch_mstep(obs)

            loglikelihood = -np.sum(np.log(self.c[:]))
            if np.abs(old_loglikelihood - loglikelihood) < eps:
                break
            old_loglikelihood = loglikelihood
            print(loglikelihood)



hmm1 = HMM(A, B, pi)
obs, state = hmm1.sample(1000)

hmm2 = HMM(eA, eB, epi)
hmm2.baum_welch(obs)

print("Actual parameters")
print(hmm1.A)
print(hmm1.B)
print(hmm1.pi)

print("Estimated parameters")
print(hmm2.A)
print(hmm2.B)
print(hmm2.pi)
