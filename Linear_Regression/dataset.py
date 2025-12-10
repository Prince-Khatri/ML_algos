import pandas as pd
import pickle as pkl
def create_dataset():
    df = pd.read_csv('/dataset/boston_dataset.csv')

    with open('/dataset/data.bin','wb+') as f:
        pkl.dump(df,f)

