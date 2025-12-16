import numpy as np

from decorators.TimeTaken import time_taken
class GradientDescend:
    """
        Gradient descend for Linear regression
    """
    def __init__(self,w,b,a,get_derivaties, get_cost):
        self.w = w
        self.b = b
        self.a = a
        self.get_derivaties = get_derivaties
        self.get_cost = get_cost

    def find_derivatives(self,X_train,y_train):
        w = self.w
        b = self.b
        return self.get_derivaties(X_train,y_train,w,b)

    def find_cost(self,X_train,y_train):
        return self.get_cost(X_train,y_train,self.w, self.b)

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