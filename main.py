from sklearn.model_selection import train_test_split
import numpy as np


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

# train-test split 80/20; not shuffled because time series
train, test = train_test_split(x, test_size=0.2, shuffle=False)

# perform quality check of the data
assert quality_check(x), "There are issues with the data: string or nan values"

# create the baseline zero model
zm = zero_model()
print("Prediction: ", zm(x))
