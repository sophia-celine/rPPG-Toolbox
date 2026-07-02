from pathlib import Path
import re
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.signal import butter

fs = 30  # adjust to your sampling frequency

folder_path = Path('/home/sophia/rPPG-Toolbox/BVPresults')

selected_methods = [
    "POS"
]

files = []

for file_path in folder_path.glob("BVP_*.txt"):
    match = re.match(r"BVP_(.*?)_(subject\d+)\.txt", file_path.name)

    if match:
        method = match.group(1)
        subject = match.group(2)

        if method in selected_methods:
            files.append((method, subject, file_path))

# Sort by method order
files.sort(key=lambda x: selected_methods.index(x[0]))

n = len(files)

if n == 0:
    print("No matching files found.")
else:
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)

    if n == 1:
        axes = [axes]

    b, a = butter(
        1,
        [0.6 / fs * 2, 3.3 / fs * 2],
        btype="bandpass"
    )

    for ax, (method, subject, file_path) in zip(axes, files):
        signal = np.loadtxt(file_path)
        signal = scipy.signal.filtfilt(b, a, np.double(signal))

        ax.plot(signal)
        ax.set_title(f"{method} - {subject}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()