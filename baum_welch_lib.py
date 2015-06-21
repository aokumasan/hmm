import numpy as np

class BaumWelch:
    # constructor
    def __init__(self, A, B, pi):
        self.A = A     # transition matrix
        self.B = B     # emission matrix
        self.pi = pi   # start prob
        self.state_num = A.shape[0]
        self.symbol_num = B.shape[1]
        self.symbol = np.arange(self.symbol_num)
        self.pA = A
        self.pB = B
        self.pPi = pi
        
        
    # initialize variables (used in M-Step)
    def init_variables(self):
        self.A_num = np.zeros((self.state_num, self.state_num))
        self.A_den = np.zeros((self.state_num, self.state_num))
        self.B_num = np.zeros((self.state_num, self.symbol_num))
        self.B_den = np.zeros((self.state_num, self.symbol_num))
        self.npi = np.zeros(self.state_num)


    # scaled forward algorithm
    def forward(self, obs):
        # length of observations
        n = len(obs)

        # init variables
        alpha = np.zeros((n, self.state_num))
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
        beta = np.zeros((n, self.state_num))

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
        for j in range(0, self.state_num):
            for k in range(0, self.symbol_num):
                self.B_num[j, k] += np.sum((obs[:] == self.symbol[k]) * self.alpha[:, j] * self.beta[:, j] / self.c[:])
                self.B_den[j, k] += np.sum(self.alpha[:, j].T * self.beta[:, j] / self.c[:])

        # calc pi
        self.npi += self.alpha[0, :] * self.beta[0, :] / self.c[0]


    # Baum Welch algorithm with Multiple Sequences of observation symbols
    def train(self, obs, delta = 1e-9, max_iter = 400):

        # init
        seq_num = obs.shape[0]
        loglik = np.zeros(seq_num)
        p_loglik = np.zeros(seq_num)

        for count in range(0, max_iter):
            if count % 10 == 0:
                print("iter: [", count, "]")
                
            self.init_variables()

            # calc alpha, beta, c each sequences
            for s in range(0, seq_num):
                s_obs = obs[s]
                # E-Step
                self.forward(s_obs)
                self.backward(s_obs)
                # M-Step
                self.maximization_step(s_obs)
                # calc log-likelihood
                loglik[s] = -np.sum(np.log(self.c[:]))

            # update parameter
            self.A = self.A_num / self.A_den.T
            self.B = self.B_num / self.B_den
            self.pi = self.npi / seq_num
            
            # convergence check
            diffA = np.power(self.A - self.pA, 2)
            diffB = np.power(self.B - self.pB, 2)
            diffPi = np.power(self.pi - self.pPi, 2)
            if (diffA < delta).all() and (diffB < delta).all() and (diffPi < delta).all():
                print("convergence !! iter = ", count)
                break


            self.pA = self.A
            self.pB = self.B
            self.pPi = self.pi

