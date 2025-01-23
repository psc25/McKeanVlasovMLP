import numpy as np
from McKV_MLP import MLP_model
import time
import os

path = os.path.join(os.getcwd(), "kuramoto\\")

dd = [10, 50, 100, 500, 1000, 5000, 10000]
runs = 10
MNmax = 5

T = 1.0
d1 = 10000

print("======================================================================")
for i in range(len(dd)):
    d = dd[i]
    mu0 = 1.0
    def mu(x, y):
        return mu0*np.sin(x - y)
    
    sigma0 = 0.3
    def sigmadiag(x, y):
        return sigma0*x
    
    mu_cost = 3*d
    sigmadiag_cost = d
        
    for MN in range(MNmax, MNmax+1):
        K = np.power(MN, MN)+1
        k = int(np.ceil(500/K))
        K1 = k*K
        tt = np.reshape(np.linspace(0.0, T, K), [-1, 1])
        tt1 = np.reshape(np.linspace(0.0, T, K1), [-1, 1])
        xi = 10.0*np.ones(d, dtype = np.float32)
            
        for r in range(1, runs+1):
            dW1 = np.random.normal(size = [K1, d1], scale = np.sqrt(T/K1))
            dW = np.add.reduceat(dW1[:, :d], np.arange(stop = K1, step = k))
            
            b = time.time()
            MLP = MLP_model(d, MN, K, T, mu, sigmadiag, mu_cost, sigmadiag_cost)
            X_mlp = MLP.compute(xi, MN, dW)
            cst = MLP.cost
            e = time.time()
            
            print("MLP performed for d = " + str(d) + ", m = " + str(MN) + "/" + str(MNmax) + ", run " + str(r) + "/" + str(runs) + ", in " + str(np.round(e-b, 1)) + "s")
            
            X_tru = np.zeros([K1, d1])
            X_tru[0] = 10.0*np.ones(d1, dtype = np.float32)
            for t in range(1, K1):
                Ecos = np.mean(np.cos(X_tru[k-1]))
                Esin = np.mean(np.sin(X_tru[k-1]))
                X_tru[t] = X_tru[t-1] + mu0*(np.sin(X_tru[k-1])*Ecos - np.cos(X_tru[k-1])*Esin)*T/K1 + sigma0*X_tru[t-1]*dW1[(t-1):t]
            
            np.savetxt(path + "mlp_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv", X_mlp)
            np.savetxt(path + "tru_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv", X_tru[::k, :d])
            np.savetxt(path + "tms_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv", [e-b])
            np.savetxt(path + "cst_" + str(dd[i]) + "_" + str(MN) + "_" + str(r) + ".csv", cst)
    
print("======================================================================")
print("MLP solutions saved")