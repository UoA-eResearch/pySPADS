import numpy as np
from scipy.optimize import fmin
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import LinearModel

# Recreation of linear regression method in matlab code - appears to be less accurate than scikit-learns implementation


def calc_sse_all_coef(theta: np.ndarray, S, PC, fit_intercept=False):
    m = np.zeros((1, PC.shape[0]))
    for j in range(PC.shape[1]):
        X1 = theta[j] * PC[:, j]
        m += X1
    if fit_intercept:
        m += theta[-1]

    d1 = np.sum((S - m) ** 2)
    d2 = np.var(S) + np.var(m) + (np.mean(S) - np.mean(m)) ** 2
    return -(1 - d1 / (d2 * len(S)))


class MReg2(LinearModel):
    def __init__(self, fit_intercept=False):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        # Initial solution from linear regression
        reg = LinearRegression(fit_intercept=self.fit_intercept).fit(X, y)
        bb = reg.coef_

        if self.fit_intercept:
            bb = np.append(bb, reg.intercept_)

        # Further optimize fit
        beta = fmin(
            calc_sse_all_coef,
            bb,
            args=(y.to_numpy(), X.to_numpy(), self.fit_intercept),
            maxiter=100000,
            maxfun=100000,
            disp=False,
        )

        if self.fit_intercept:
            self.coef_ = beta[:-1]
            self.intercept_ = beta[-1]
        else:
            self.coef_ = beta
            self.intercept_ = 0

        return self
