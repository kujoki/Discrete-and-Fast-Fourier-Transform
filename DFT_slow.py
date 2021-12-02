import time
import numpy as np
start_time = time.time()

def DFT_slow(x): #простое Фурье (дискретное)
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)
x = np.random.random(1024)
x_f = DFT_slow(x)
print("--- %s seconds ---" % (time.time() - start_time))
