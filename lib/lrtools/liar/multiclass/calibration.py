"""
A calibrator takes probabilities as input, and returns calibrated probabilities
as output.

X is a numpy array with dimensions m, n, where
  - m is the number of instances and
  - n is the number of classes
  - X[m,n] is the probability of instance m being of class n

y is an array of length m, where
  - y[m] is the class label of instance m
"""

import logging
import math

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.neighbors import KernelDensity

import numpy as np


LOG = logging.getLogger(__name__)


class DummyCalibrator(BaseEstimator, TransformerMixin):
    """
    Takes probabilities as input, and returns calibrated probabilities as output.
    """

    def fit(self, X, y, **fit_params):
        self.classes_ = np.unique(y)
        return self

    def transform(self, X):
        return X


class KDECalibrator(BaseEstimator, TransformerMixin,):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses kernel density estimation (KDE) for interpolation.
    """

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def bandwidth_silverman(self, X):
        """
        Estimates the optimal bandwidth parameter using Silverman's rule of
        thumb.
        """
        assert len(X) > 0

        std = np.std(X)
        if std == 0:
            # can happen eg if std(X) = 0
            LOG.info('found a silverman bandwidth of 0 (using dummy value)')
            std = 1

        v = math.pow(std, 5) / len(X) * 4. / 3
        return math.pow(v, .2)

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        self._kde_target = []
        self._kde_other = []

        for cls in range(len(self.classes_)):
            X_target = X[y==self.classes_[cls],cls]
            X_other = X[y!=self.classes_[cls],cls]

            if self.bandwidth is None:
                bandwidth_target = self.bandwidth_silverman(X_target)
                bandwidth_other = self.bandwidth_silverman(X_other)
            else:
                bandwidth_target = self.bandwidth
                bandwidth_other = self.bandwidth

            self._kde_target.append(KernelDensity(kernel='gaussian', bandwidth=bandwidth_target).fit(X_target.reshape(-1,1)))
            self._kde_other.append(KernelDensity(kernel='gaussian', bandwidth=bandwidth_other).fit(X_other.reshape(-1,1)))

        return self

    def transform(self, X):
        X_out = []
        for cls in range(len(self.classes_)):
            X_target = X[:,cls]
            X_other = X[:,cls]
            #X_other = np.sum(np.delete(X, cls, axis=1), axis=1)

            p_target = np.exp(self._kde_target[cls].score_samples(X_target.reshape(-1,1)))
            p_other = np.exp(self._kde_other[cls].score_samples(X_other.reshape(-1,1)))

            X_out.append(p_target / (p_target + p_other))

        return np.stack(X_out, axis=1)
