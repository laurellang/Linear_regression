import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import time

class ManualScaler(TransformerMixin, BaseEstimator):
    def fit(self,X,y=None):
        self.mu_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self,X):
        return (X-self.mu_) / self.std_





class ManualLinearRegression(RegressorMixin, BaseEstimator):
    def __init__(self, learning_rate, n_iterations, l2_strength=0.0):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l2_strength = l2_strength

    def compute_cost(self,X,y):
        m,n = X.shape
        total_cost = 0
        for i in range(m):
            err = (np.dot(self.w_,X[i])+self.b_-y[i])*(np.dot(self.w_,X[i])+self.b_-y[i])
            total_cost += err
        total_cost /= (2*m)

        #L2
        l2 = self.l2_strength/(2*m)*np.sum(self.w_ ** 2)


        return total_cost + l2

    def compute_gradient(self,X,y):
        m,n = X.shape
        err = X @ self.w_ + self.b_ - y
        dj_dw = (X.T @ err) / m
        dj_db = np.mean(err)

        dj_dw += (self.l2_strength / m) * self.w_

        return dj_dw,dj_db
    # def fit(self,X,y):
    #     self.w_ = np.zeros(X.shape[1])
    #     self.b_ = 0.0
    #     for i in range(self.n_iterations):
    #         dj_dw,dj_db = self.compute_gradient(X,y)
    #         self.w_ -= self.learning_rate * dj_dw
    #         self.b_ -= self.learning_rate * dj_db
    #
    #     return self

    def fit(self, X, y):
        self.w_ = np.zeros(X.shape[1])
        self.b_ = 0.0
        start_time = time.perf_counter()
        for i in range(self.n_iterations):
            dj_dw, dj_db = self.compute_gradient(X, y)

            self.w_ -= self.learning_rate * dj_dw
            self.b_ -= self.learning_rate * dj_db

            #进度信息
            completed = i + 1

            if completed == 1 or completed % 100 == 0 or completed == self.n_iterations:
                cost = self.compute_cost(X, y)
                elapsed = time.perf_counter() - start_time
                remaining = elapsed / completed * (self.n_iterations - completed)

                print(
                    f"迭代 {completed}/{self.n_iterations} | "
                    f"损失 {cost:.8f} | "
                    f"已用 {elapsed:.1f} 秒 | "
                    f"预计剩余 {remaining:.1f} 秒",
                    flush=True,
                )

        return self

    def predict(self,X):
        m = X.shape[0]
        y_pred = np.zeros(m)
        for i in range(m):
            y_pred[i] = np.dot(self.w_,X[i]) + self.b_
        return y_pred








