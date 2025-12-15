from Linear_Regression.LinearRegression import LinearRegression
from Linear_Regression.Normalise import Normalise
import numpy as np
import os
import pickle as pkl
saved_path = "Linear_Regression/dataset/data.bin"

def train():
    # Load dataset from saved files
    print("Loading data to train:")
    with open(saved_path,'rb') as f:
        data = pkl.load(f)
        X_train = data['X_train']
        y_train = data['y_train']
        X_cv = data['X_cv']
        y_cv = data['y_cv']
        X_test = data['X_test']
        y_test = data['y_test']
    print("Setting up model config:")
    
    norm_model = Normalise()
    norm_model.compile(X_train)
    X_train = norm_model.fit(X_train)
    X_test = norm_model.fit(X_test)

    model = LinearRegression(X_train,y_train,X_cv,y_cv,normaliser = norm_model)


    model.compile(learning_rate = 0.001)
    print("Compiled.")
    model.fit()
    print("trained")
    
    mse,rmse = model.get_mse_rmse(X_test,y_test)
    print(f"Test mse:{mse}\nTest rmse:{rmse}")
    version = len(os.listdir('Linear_Regression/models/'))
    with open(f'Linear_Regression/models/lr_v{version}.bin','wb') as f:
        pkl.dump(model,f)
    

if __name__ == '__main__':
    train()
