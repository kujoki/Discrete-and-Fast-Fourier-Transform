import time
import numpy as np
from scipy.fft import fft, ifft
start_time = time.time()

x = np.random.random(1024)
x_f = fft(x)

print("--- %s seconds ---" % (time.time() - start_time))

 # как красив Татарстан когда мы используем scipy
 # --- 0.00016021728515625 seconds ---
