import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import dataset
import numpy as np

df = dataset.get_flights_dataset()

passengers = df["passengers"]
avg = passengers - passengers.mean()
fft = np.fft.fft(avg)

abs_fft = np.abs(fft)
timestep = 1 # 1 datapoint per month
fft_freq = np.fft.fftfreq(passengers.size, d=timestep)

# plt.plot(2*np.abs(fft_freq), abs_fft)
# peaks, _ = find_peaks(abs_fft)
# lt.plot(2*np.abs(fft_freq)[peaks], abs_fft[peaks], "x")
# plt.show()

# plt.plot(range(len(avg)), avg)
# plt.axvline(24)
# plt.show()
