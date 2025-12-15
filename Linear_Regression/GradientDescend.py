import numpy as np

from decorators.TimeTaken import time_taken
class GradientDescend:
    """
        Gradient descend for Linear regression
    """
    def __init__(self,w,b,a):
        self.w = w
        self.b = b
        self.a = a

    def find_derivatives(self,X_train,y_train):
        w = self.w
        b = self.b
        m = len(X_train)

        y_pred = X_train @ w + b
        error = y_pred - y_train
        dJ_dw = (X_train.T @ error)/m
        dJ_db = error.sum()/m
        
        return dJ_dw,dJ_db

    def find_cost(self,X_train,y_train):
        m = len(X_train)
        cost = 0
        y_pred = X_train @ self.w + self.b
        cost = np.power(y_pred-y_train,2).sum()
        cost/= 2*m
        return cost

    def find_w_b(self, X_train, y_train, max_range=1_000_000, threshold=1e-5):

        prev_cost = self.find_cost(X_train,y_train)

        for i in range(max_range):

            dJ_dw, dJ_db = self.find_derivatives(X_train,y_train)
            self.w = self.w - dJ_dw*self.a
            self.b = self.b - dJ_db*self.a

            cost = self.find_cost(X_train,y_train)

            if abs(cost-prev_cost)/(prev_cost+1e-8)< threshold:
                break
            prev_cost = cost

            if i%10000 == 0:
                print(f"cost:{cost}")
        return self.w,self.b