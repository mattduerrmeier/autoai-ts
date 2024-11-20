import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import dataset
import numpy as np

df = dataset.get_flights_dataset()

passengers = df["passengers"]
avg = passengers - passengers.mean()
fft = np.fft.fft(avg)

timestep = 1 # because we have datapoint every month
fft_freq = np.fft.fftfreq(passengers.size, d=timestep)

plt.plot(2* np.abs(fft_freq), np.abs(fft))
plt.show()
