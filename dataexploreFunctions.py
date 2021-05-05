import pandas as pd
import numpy as np
import math


class Smoothing:
    def __init__(self, dataframe, window_half = 1, order=1, deriv=1):
        self.dataframe = dataframe
        self.window_half = window_half
        self.order = order
        self.deriv = deriv

        y_ = self.dataframe.copy()
        B = np.mat(
        [[k**i for i in range(self.order + 1)] for k in range(-1 * self.window_half, self.window_half + 1)]
        )
        M = math.factorial(self.deriv) * np.linalg.pinv(B).A[deriv]
        y_ = np.convolve(M, y_, mode = 'valid')

        return y_



class BaseLineCorrect:
    def __init__(self, dataframe, wavelength):
        self.dataframe = dataframe
        self.wavelength = wavelength

    def spBaseLine(self):
        converted_lam = "{}".format(self.wavelength)
        diff = pd.DataFrame(0 - self.dataframe[converted_lam])
        y_diff = np.add(self.dataframe, diff.to_numpy())

        return y_diff
