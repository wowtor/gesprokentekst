import collections
import logging

import numpy as np


LOG = logging.getLogger(__name__)


LR = collections.namedtuple('LR', ['lr', 'p_target', 'p_other', 'target'])


class CalibratedScorer:
    def __init__(self, scorer, calibrator, fit_calibrator=False):
        self.scorer = scorer
        self.calibrator = calibrator
        self.fit_calibrator = fit_calibrator

    def fit(self, X, y):
        self.scorer.fit(X, y)
        if self.fit_calibrator:
            p = self.scorer.predict_proba(X)
            self.calibrator.fit(p, y)

    def predict_proba(self, X, y):
        X = self.scorer.predict_proba(X)
        return self.calibrator.transform(X)


def lr(X, y):
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    p_target = X[:,y]
    p_other = (1 - X[:,y])
    return np.stack([p_target / p_other, p_target, p_other])


def cllr(X, y):
    """
    Calculates a log likelihood ratio cost (C_llr) for a series of likelihood
    ratios.

    Parameters
    ----------
    lrs : list of LR objects
    target : ...

    Returns
    -------
    LrStats
        Likelihood ratio statistics.
    """
    def avg(*args):
        return sum(args) / len(args)

    class_size = dict((cls, np.count_nonzero(y==cls)) for cls in np.unique(y))
    class_weight = np.vectorize(lambda c: 1/class_size[c])(y)

    p_target = X[np.arange(len(X)),y]

    with np.errstate(divide='ignore'):
        return np.average(np.log2(1 + (1 - p_target) / p_target), weights=class_weight)
