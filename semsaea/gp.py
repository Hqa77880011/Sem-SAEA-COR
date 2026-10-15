import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class GaussianProcess:
    def __init__(self, prior=None, strength=0):
        self.prior, self.strength = prior, strength

    @staticmethod
    def kernel(a, b, theta):
        distance = np.sqrt(np.sum(((a[:, None] - b[None]) / np.exp(theta[:-2])) ** 2, axis=-1) + 1e-20)
        root = np.sqrt(5) * distance
        return np.exp(2 * theta[-2]) * (1 + root + 5 * distance ** 2 / 3) * np.exp(-root)

    def fit(self, x, y):
        self.x = np.asarray(x)
        self.mean, self.scale = float(np.mean(y)), max(float(np.std(y)), 1e-8)
        target = (np.asarray(y) - self.mean) / self.scale
        dimensions = self.x.shape[1]

        def objective(theta):
            covariance = self.kernel(self.x, self.x, theta) + np.eye(len(x)) * (np.exp(2 * theta[-1]) + 1e-8)
            factor = cho_factor(covariance, lower=True)
            alpha = cho_solve(factor, target)
            loss = 0.5 * target @ alpha + np.log(np.diag(factor[0])).sum() + 0.5 * len(x) * np.log(2 * np.pi)
            if self.prior is not None:
                loss += 0.5 * self.strength * np.sum((theta[:-2] - self.prior) ** 2)
            return float(loss)

        initial = np.r_[np.full(dimensions, np.log(0.3)), 0, np.log(0.01)]
        fitted = minimize(objective, initial, method="L-BFGS-B", bounds=[(-5, 2)] * dimensions + [(-4, 4), (-10, -1)])
        if not np.isfinite(fitted.fun):
            raise ArithmeticError("GP fit did not produce a finite likelihood")
        self.theta = fitted.x
        covariance = self.kernel(self.x, self.x, self.theta) + np.eye(len(x)) * (np.exp(2 * self.theta[-1]) + 1e-8)
        self.factor = cho_factor(covariance, lower=True)
        self.alpha = cho_solve(self.factor, target)
        inverse_diagonal = np.diag(cho_solve(self.factor, np.eye(len(x))))
        variance = 1 / inverse_diagonal
        residual = self.alpha / inverse_diagonal
        self.loo_nrmse = float(np.sqrt(np.mean(residual ** 2)))
        self.loo_r2 = float(1 - np.sum(residual ** 2) / max(np.sum(target ** 2), 1e-12))
        self.loo_nlpd = float(np.mean(0.5 * (np.log(2 * np.pi * variance) + residual ** 2 / variance)))
        return self

    def predict(self, x):
        cross = self.kernel(np.asarray(x), self.x, self.theta)
        mean = self.mean + self.scale * (cross @ self.alpha)
        variance = np.exp(2 * self.theta[-2]) - np.sum(cross * cho_solve(self.factor, cross.T).T, axis=1)
        return mean, self.scale * np.sqrt(np.maximum(variance, 1e-12))
