from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# TODO: raise errors instead?
# Check for negative value
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

def get_look_back_window_length(x, col_idx=0):
    """
    Computes the look back window length for the input dataset.
    By default, it is assumed that the timestamp is the first column of the 2D array.
    """
    # TODO: implement timestamp based assessment with temporal frequency

    # value index assessment
    # 1. zero-crossing
    value_col = x[:, -1]
    value_col -= np.mean(value_col);

    # the bit sign is an array of booleans; do a diff (x[i+1] - x[i]) and find indices of true
    zero_crossing_idxs = np.nonzero(np.diff(np.signbit(value_col)))[0]
    zero_crossing_mean = np.mean(zero_crossing_idxs)

    # TODO: 2. spectral analysis

    # select look-back window
    looks_backs = [candidate for candidate in [zero_crossing_mean] if candidate > len(x)]

    look_back = 0

    if len(look_backs) > 1:
        # which look back to keep?
        look_back = select_look_back(look_backs)
    elif len(look_backs) == 1:
        look_back = look_backs[0]
    else:
        look_back = 8 # default value

    return look_back


def select_look_back(look_backs):
    # TODO: implement the look back selection as described in the paper
    return look_backs[0]


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
