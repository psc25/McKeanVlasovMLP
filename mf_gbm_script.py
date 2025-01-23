import numpy as np
from McKV_MLP import MLP_model
import time
import os

path = os.path.join(os.getcwd(), "mf_gbm\\")

dd = [10, 50, 100, 500, 1000, 5000, 10000]
runs = 10
MNmax = 5

T = 1.0

print("======================================================================")
for i in range(len(dd)-1, len(dd)):
    d = dd[i]
    mu0 = -0.1
    def mu(x, y):
        return 0.5*mu0*(x + y)
    
    sigma0 = 0.2
    def sigmadiag(x, y):
        return 0.5*sigma0*(x + y)
    
    mu_cost = 2+d
    sigmadiag_cost = 2+d
    for MN in range(1, MNmax+1):
        K = np.power(MN, MN)+1
        k = int(np.ceil(500/K))
        K1 = k*K
        tt = np.reshape(np.linspace(0.0, T, K), [-1, 1])
        tt1 = np.reshape(np.linspace(0.0, T, K1), [-1, 1])
        xi = 30.0*np.ones(d, dtype = np.float32)
        for r in range(1, runs+1):
            dW1 = np.random.normal(size = [K1, d], scale = np.sqrt(T/K1))
            dW = np.add.reduceat(dW1, np.arange(stop = K1, step = k))
            
            b = time.time()
            MLP = MLP_model(d, MN, K, T, mu, sigmadiag, mu_cost, sigmadiag_cost)
            X_mlp = MLP.compute(xi, MN, dW)
            cst = MLP.cost
            e = time.time()
            
            print("MLP performed for d = " + str(d) + ", m = " + str(MN) + "/" + str(MNmax) + ", run " + str(r) + "/" + str(runs) + ", in " + str(np.round(e-b, 1)) + "s")
            
            X_tru = np.zeros([K1, d])
            X_tru[0] = xi
            for t in range(1, K1):
                Exk1 = np.exp(mu0*tt1[t-1])*xi
                X_tru[t] = X_tru[t-1] + mu(X_tru[t-1], Exk1)*(T/K1) + sigmadiag(X_tru[t-1], Exk1)*dW1[(t-1):t]
            
            np.savetxt(path + "mlp_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv", X_mlp)
            np.savetxt(path + "tru_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv", X_tru[::k])
            np.savetxt(path + "tms_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv", [e-b])
            np.savetxt(path + "cst_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv", [cst])
    
print("======================================================================")
print("MLP solutions saved")