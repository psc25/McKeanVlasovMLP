import numpy as np

class MLP_model():
    def __init__(self, d, M, K, T, mu, sigmadiag, mu_cost, sigmadiag_cost, dtype = np.float32):
        self.d = d
        self.M = M
        self.K = K
        self.T = T
        self.dt = T/K
        self.tt = np.reshape(np.linspace(0.0, 1.0, K), [-1, 1])
        self.mu = mu
        self.sigmadiag = sigmadiag
        self.mu_cost = int(mu_cost)
        self.sigmadiag_cost = int(sigmadiag_cost)
        self.dtype = dtype
        self.cost = np.array([0], dtype = np.int64)
        
    def compute(self, xi, N, dW):
        if N == 0:
            return np.zeros([self.K, self.d], dtype = self.dtype)
        
        X = xi
        X = X + self.tt*np.expand_dims(self.mu(np.zeros(self.d), np.zeros(self.d)), 0)
        self.cost = self.cost + 2*self.K*self.d + self.mu_cost
        X[1:] = X[1:] + self.sigmadiag(np.zeros(self.d), np.zeros(self.d))*np.cumsum(dW[:-1], axis = 0)
        self.cost = self.cost + 2*self.K*self.d + self.sigmadiag_cost
        for l in range(1, N):
            mnl = np.power(self.M, N-l)
            for k in range(mnl):
                dW2 = np.random.normal(size = [self.K, self.d], scale = np.sqrt(self.dt)).astype(dtype = self.dtype)
                self.cost = self.cost + self.K*self.d
                x1 = self.compute(xi, l, dW)
                x2 = self.compute(xi, l, dW2)
                x3 = self.compute(xi, l-1, dW)
                x4 = self.compute(xi, l-1, dW2)
                int_dW = np.zeros([self.K, self.d])
                for t in range(1, self.K):
                    int_dW[t] = int_dW[t-1] + (self.sigmadiag(x1[t-1], x2[t-1]) - self.sigmadiag(x3[t-1], x4[t-1]))/mnl*dW[t-1]
                    
                self.cost = self.cost + self.K*(3*self.d + 2*self.sigmadiag_cost + 1)
                    
                X = X + int_dW
                self.cost = self.cost + self.K*self.d
                
                u = np.random.uniform()
                self.cost = self.cost + 1
                for t in range(1, self.K):
                    r = int(np.floor(self.K*u*self.tt[t]))
                    X[t] = X[t] + self.tt[t]*(self.mu(x1[r], x2[r])-self.mu(x3[r] ,x4[r]))/mnl
                    
                self.cost = self.cost + self.K*(3*self.d + 2*self.mu_cost + 1)
                    
        return X