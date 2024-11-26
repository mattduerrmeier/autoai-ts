import pandas as pd
import numpy as np
from t_daub import TDaub
import dataset
import pipeline
import time

d = {"size": [], "mean_time": []}

for i in range(0, 4):
    l = []
    size = 10**i * 100
    print(f"size: {size}")
    d["size"].append(size)

    arr = dataset.get_cosine_function(time=size)
    X, y = dataset.to_supervised(arr)

    window_length = size // 10

    for j in range(10):
        pipelines = pipeline.create_pipelines(random_state=42+j)
        tdaub = TDaub(pipelines)

        start = time.time()
        tdaub.fit(
            X, y,
            allocation_size=window_length,
            verbose=False
        )
        stop = time.time()

        time_diff = stop - start
        l.append(time_diff)

    d["mean_time"].append(np.mean(l))

df = pd.DataFrame(d)
print(df)
df.to_csv("bench.csv", index=False)
