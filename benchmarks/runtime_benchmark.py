import pandas as pd
import numpy as np
from autoai_ts import AutoAITS
from pipeline import create_pipelines
import dataset
import time
import warnings

warnings.filterwarnings("ignore")

d: dict[str, list[float]] = {"size": [], "mean_time": []}

for i in range(0, 4):
    time_diffs = []
    size = 10 ** (i + 2)  # 100, 1'000, 10'000, 100'000
    print(f"size: {size}")
    d["size"].append(size)

    arr = dataset.get_cosine_function(time=size)
    X, y = dataset.to_supervised(arr)

    window_length = size // 10

    for j in range(10):
        print(".", end="", flush=True)
        pipelines = create_pipelines(random_state=42 + j, contains_neg_values=True)
        model = AutoAITS(pipelines)

        start = time.time()
        model.t_daub(X, y, allocation_size=window_length, verbose=False)
        stop = time.time()
        time_diff = stop - start
        time_diffs.append(time_diff)

    d["mean_time"].append(float(np.mean(time_diffs)))
    print(" -> Done!")

df = pd.DataFrame(d)
print(df)
df.to_csv("bench.csv", index=False)
