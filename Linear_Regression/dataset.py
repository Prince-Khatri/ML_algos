import pandas as pd
import pickle as pkl
import os
def create_dataset(path):
    """
        creates an binary file with dataset saved in it.
    """
    df = pd.read_csv(path)
    saved_path = f"Linear_Regression/dataset/data{len(os.listdir())}.bin"
    with open(saved_path,'wb+') as f:
        pkl.dump(df,f)
    
    return saved_path, df.columns.tolist()

