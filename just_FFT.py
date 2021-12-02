import time
import numpy as np
start_time = time.time()

def FFT(x):
    # БПФ
    x = np.asarray(x, dtype=float)  # Мелкая копия (вещественная)
    N = x.shape[0]  # Сначала 1024, потом на 512 и т.д.

    if N % 2 > 0:
        raise ValueError("Степень 2") #главная фишка
    elif N <= 16:
        n = np.arange(N) # [0 1 2 3 ... 31]
        k = n.reshape((N, 1)) # [[]]
        M = np.exp(-2j * np.pi * k * n / N) # финал
        return np.dot(M, x)
    else:
        X_even = FFT(x[::2]) #   Старт с 0, шаг 2
        X_odd = FFT(x[1::2]) # Старт с 1, шаг 2
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])
x = np.random.random(1024)
x_f = FFT(x)
print("--- %s seconds ---" % (time.time() - start_time))

