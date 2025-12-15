import pandas as pd
import numpy as np
from decorators.TimeTaken import time_taken
from decorators.PrettyPrint import pretty_print,pretty_header,pretty_footer
import matplotlib.pyplot  as plt 
import pickle as pkl
import os

dataset_loc = 'Linear_Regression/dataset/boston_dataset.csv'
saved_path = f"Linear_Regression/dataset/data.bin"
norm_model = "Linear_Regression/dataset/norm_model.bin"

@pretty_header
def visualise_dataset(df):
    print(f"        Shape:{df.shape}\n        Size:{df.size}")
    columns = df.columns.tolist()
    f = plt.figure(figsize = (10,10))
    for i,j in enumerate(columns):
        
        plt.subplot(3,5,i+1)
        plt.hist(df[j].to_numpy(),bins=50)

        plt.title(j)
    plt.savefig('Linear_Regression/dataset/hist_data.png')
    print("Saving the histogram of all columns")

@pretty_header
def create_dataframe(path):
    """
        read a csv file and generate a df
    """
    print("Creating Dataframe:")
    df = pd.read_csv(path)
    print("Sucessfully Created")
    return df

@pretty_header
def create_set(df):
    print("Creating data to train, cv and test in model...")
    columns = df.columns.tolist()
    X = df[columns[0:-1]]
    y = df[columns[-1]]
    return X.to_numpy(),y.to_numpy()

@pretty_header
def split(X,y,train=0.7,cv=0.15,test=0.15,seed=12):
    print("Creating test cv train split")
    size = len(X)
    # get shuffered indicies
    np.random.seed(seed)
    indicies = np.random.permutation(size)
    mask_train = int(size*train)
    mask_cv = int(size*cv)

    X_train = X[indicies[0:mask_train]]
    y_train = y[indicies[0:mask_train]]

    X_cv = X[indicies[mask_train:mask_train+mask_cv]]
    y_cv = y[indicies[mask_train:mask_train+mask_cv]]

    X_test = X[indicies[mask_train+mask_cv::]]
    y_test = y[indicies[mask_train+mask_cv::]]

    return X_train,y_train,X_cv,y_cv,X_test,y_test

@pretty_footer
@time_taken 
@pretty_print

def main():
    
    df = create_dataframe(dataset_loc)
    
    
    # No need to clean as dataset is clean already,
    # print(df.describe(),df.isna().sum(),df.duplicated().sum())

    # if so then you can execute the below commands
    # df.dropna()
    # df.drop_duplicates()
    

    visualise_dataset(df)
    

    # Creating data to train, cv and test in model...
    X,y = create_set(df)

    # Creating test cv train split
    X_train,y_train,X_cv,y_cv,X_test,y_test = split(X,y,seed=42)
    data = {'X_train':X_train,
        'y_train':y_train,
        'X_cv':X_cv,
        'y_cv':y_cv,
        'X_test':X_test,
        'y_test':y_test
    }
    
    with open(saved_path,'wb+') as f:
        pkl.dump(data,f)
    
    print(f"Data sucessfully saved in {saved_path}")

    
if __name__ == '__main__':
    main()


