import numpy as np

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

class BaumWelch:
    # constructor
    def __init__(self, A, B, pi):
        self.A = A     # transition matrix
        self.B = B     # emission matrix
        self.pi = pi   # start prob

        
    def init_variables(self):
        # use in M-Step
        self.A_num = np.zeros((state_num, state_num))
        self.A_den = np.zeros((state_num, state_num))
        self.B_num = np.zeros((state_num, symbol_num))
        self.B_den = np.zeros((state_num, symbol_num))
        self.mpi = np.zeros(state_num)


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
        beta[n-1, :] = self.c[n-1]

        # induction
        for t in range((n-1), 0, -1):
            beta[t-1, :] = np.dot(self.A, (self.B[:, obs[t]] * beta[t, :]))
            beta[t-1, :] = self.c[t-1] * beta[t-1, :]

        self.beta = beta


    # M-Step
    def maximization_step(self, obs):
        # length of observations
        n = len(obs)

        # calc A
        for t in range(0, n-1):
            self.A_num += self.alpha[t, :][:, np.newaxis] * self.A * self.B[:, obs[t+1]] * self.beta[t+1, :]
            self.A_den += self.alpha[t, :] * self.beta[t, :] / self.c[t]

        # calc B
        for j in range(0, state_num):
            for k in range(0, symbol_num):
                self.B_num[j, k] += np.sum((obs[:] == symbol[k]) * self.alpha[:, j] * self.beta[:, j] / self.c[:])
                self.B_den[j, k] += np.sum(self.alpha[:, j].T * self.beta[:, j] / self.c[:])

        # update pi
        self.mpi += self.alpha[0, :] * self.beta[0, :] / self.c[0]



    # Baum Welch algorithm with Multiple Sequences of observation symbols
    def baum_welch_m(self, obs, delta = 1e-9, max_iter = 400):

        seq_num = obs.shape[0]
        loglik = np.zeros(seq_num)
        p_loglik = np.zeros(seq_num)
        
        for count in range(0, max_iter):
            self.init_variables()
            
            for s in range(0, seq_num):
                s_obs = obs[s]
                self.forward(s_obs)
                self.backward(s_obs)
                self.maximization_step(s_obs)
                loglik[s] = -np.sum(np.log(self.c[:]))

            self.A = self.A_num / self.A_den.T
            self.B = self.B_num / self.B_den
            self.pi = self.mpi / seq_num
            
            if all(np.abs(p_loglik - loglik) < np.array([delta]*seq_num)):
                break

            print("iter: [", count, "] , log-likelihood(min): [", np.min(loglik), "]")

            p_loglik = loglik.copy()

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

o1, s = simulate(300)
o2, s = simulate(50)
o3, s = simulate(100)

hmm = BaumWelch(eA, eB, epi)
obs = np.array([o1, o2, o3])
print(obs)
hmm.baum_welch_m(obs, 1e-9, 10000)

print("Actual parameters")
print(A)
print(B)
print(pi)

print("Estimated parameters")
print(hmm.A)
print(hmm.B)
print(hmm.pi)
