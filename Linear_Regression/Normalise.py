import numpy as np
import pickle as pkl
class Normalise:

    def __init__(self):
        pass

    def fit(self,X):
        return (X-self.mean)/self.std

        
    def compile(self, X_train):
        # Mean of each column
        self.mean = X_train.mean(axis=0)
        self.std = X_train.std(axis=0)


    def get_w_b(self, w_norm,b_norm):
        w = w_norm/self.std
        b = b_norm - np.dot(w,self.mean)
        return w,b
    
    def deNormalise(self,X):
        return X*self.std+self.mean

