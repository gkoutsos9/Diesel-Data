import pandas as pd
import numpy as np


y = pd.read_csv('diesel_prop.csv', index_col=0)
x = pd.read_csv('diesel_spec.csv', index_col=0)

x.T.plot(legend=False)


def baseline_correct(y, lam):
    converted_lam = "{}".format(lam)
    diff = pd.DataFrame(0 - y[converted_lam])
    y_diff = np.add(y, diff.to_numpy())
    return y_diff

new = baseline_correct(x, 1150)

new.T.plot(legend=False)
