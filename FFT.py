import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
start_time = time.time()

def DFT_slow(x): #простое Фурье (дискретное)
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

x = np.random.random(1024)
n = 2
r = [x[i:n+i] for i in range(0, len(x), n)]
print (r)
r_1 = []
r_2 = []
for i in range(0, len(r)):
    k = r[i]
    r_1.append(k[0])
    r_2.append(k[1])

plt.figure(1)
plt.scatter(r_1, r_2) #поменять местами
plt.show()


x_f = DFT_slow(x)

#print (x_f)

#timelap = time.time() - start_time
#print("--- %s seconds ---" % (time.time() - start_time))

plt.figure(2)
plt.scatter(x_f.imag, x_f.real)
plt.title('Дискретное преобразование')
plt.show()

def FFT(x):
    # БПФ
    x = np.asarray(x, dtype=float)  # Мелкая копия
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2") #главная фишка
    elif N <= 32:
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])          # Начало от 0, 2 - интервал
        X_odd = FFT(x[1::2])          # Начиная с 1, 2 - интервал
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])
x_f = FFT(x)

#print (x_f)
plt.figure(2)
plt.scatter(x_f.imag, x_f.real) #поменять местами
plt.title('Быстрое преобразование')
plt.show()

x_f = fft(x)
#print (x_f)
plt.figure(3)
plt.scatter(x_f.imag, x_f.real)
plt.show()

#print("--- %s seconds ---" % (time.time() - start_time - timelap))
