from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# TODO: raise errors instead?
def quality_check(x):
    """
    Verifies that the data has no nan values and on strings.
    Return False if there are issues with the data.
    """
    # input array should not contain strings
    status: bool = False

    # string: return false
    if x.dtype.type == np.object_:
        return False

    # no nan values
    if np.isnan(np.sum(x)) == np.True_:
        return False

    return True

def zero_model():
    """
    Returns the zero model, used as baseline by AutoAI-TS.
    This model always predicts the most recent value of the time serie.
    """
    return lambda x: x[-1]


x = np.random.rand(5, 3)

df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv')
# convert months to numbers since our pipeline requires number only
month_map = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
df["month"] = df["month"].map(month_map)

# train-test split 80/20; not shuffled because time series
train, test = train_test_split(df.to_numpy(), test_size=0.2, shuffle=False)

# perform quality check of the data
assert quality_check(train), "There are issues with the data: string or nan values"

# create the baseline zero model
zm = zero_model()
print("Prediction: ", zm(train))
