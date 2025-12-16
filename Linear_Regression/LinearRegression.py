from Linear_Regression.GradientDescend import GradientDescend
import numpy as np

from decorators.TimeTaken import time_taken
class LinearRegression():
    """
        Machine Learnign algorithm to predict real values.
        w=0
        b=0
        y_pred = np.dot(w,x) + b
        cost function J(w,b) = sum(y_pred[i]-y[i])**2/2m
    """

    def __init__(self,X_train,y_train,X_test,y_test,normaliser):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.normaliser = normaliser



    @time_taken
    def fit(self):
        self.w, self.b = self.grad_desc.find_w_b(self.X_train,self.y_train)

        
        self.test_mse,self.test_rmse = self.get_mse_rmse(self.X_test,self.y_test)
        self.train_mse,self.train_rmse = self.get_mse_rmse(self.X_train,self.y_train)

        self.w, self.b = self.normaliser.get_w_b(self.w,self.b)
        self.summary()

    def get_deratives(self,X_train,y_train,w, b):
        m = len(X_train)
        y_pred = X_train @ w + b
        error = y_pred - y_train
        dJ_dw = (X_train.T @ error)/m
        dJ_db = error.sum()/m

        return dJ_dw,dJ_db

    def get_cost(self, X_train, y_train, w, b):
        m = len(X_train)
        cost = 0
        y_pred = X_train @ w + b
        cost = np.power(y_pred-y_train,2).sum()
        cost/= 2*m
        return cost

    def get_mse_rmse(self,X,y):
        m = len(X)
        y_pred = self.predict(X)
        mse = (np.power(y_pred-y,2)).sum()/m
        rmse = np.sqrt(mse)
        return mse,rmse

    def predict(self,X_test):
        return  X_test @ self.w +self.b
        

    def compile(self,learning_rate=0.001):
        self.learning_rate=learning_rate
        self.w= np.zeros(self.X_train[0].shape)
        self.b= 0
        self.grad_desc = GradientDescend(self.w,self.b,self.learning_rate,self.get_deratives,self.get_cost)

    def summary(self):
        print(f"\tw:{self.w}\n\tb:{self.b}")
        print(f"\tTrain MSE Error:{self.train_mse}\n\tTest MSE Error:{self.test_mse}")
        print(f"\tTrain RMSE Error:{np.sqrt(self.train_mse)}\n\tTest RMSE Error:{self.test_rmse}")



