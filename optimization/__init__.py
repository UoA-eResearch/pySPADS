import numpy as np
from scipy.optimize import fmin
from sklearn.linear_model import LinearRegression

# Recreation of linear regression method in matlab code - appears to be less accurate than scikitlearns implementation


def calc_SSE_all_Coef(theta: np.ndarray, S, PC, fit_intercept=False):
    m = np.zeros((1, PC.shape[0]))
    for j in range(PC.shape[1]):
        X1 = theta[j] * PC[:, j]
        m += X1
    if fit_intercept:
        m += theta[-1]

    d1 = np.sum((S - m) ** 2)
    d2 = np.var(S) + np.var(m) + (np.mean(S) - np.mean(m)) ** 2
    return -(1 - d1 / (d2 * len(S)))


def mreg2(y, X, fit_intercept=False):
    reg = LinearRegression(fit_intercept=fit_intercept).fit(X, y)
    bb = reg.coef_
    if fit_intercept:
        bb = np.append(bb, reg.intercept_)

    beta = fmin(calc_SSE_all_Coef, bb, args=(y.to_numpy(), X.to_numpy(), fit_intercept),
                maxiter=100000, maxfun=100000,
                disp=False)

    if fit_intercept:
        return beta[:-1], beta[-1]
    else:
        return beta, 0
