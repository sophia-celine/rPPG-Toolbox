import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy
from scipy.signal import butter

fs = 30


# bvp_data = '/home/sophia/rPPG-Toolbox/BVPresults/BVP_GREEN_1_0.txt'
bvp_data = '/home/sophia/rPPG-Toolbox/BVPresults/BVP_POS_0_0.txt'

sinal = np.loadtxt(bvp_data)
metodo = Path(bvp_data).name.split("BVP_")[1].split("_0_0")[0]
[b, a] = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
sinal = scipy.signal.filtfilt(b, a, np.double(sinal))

plt.plot(sinal)
plt.title(metodo)
plt.show()