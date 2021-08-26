# %%
# File for implementing the logic for dividing the training dataset in stratified_kfolds
import config
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def generate_kfolds(k_folds):
    # Reading the data
    df_train = pd.read_csv(config.TRAIN_DATA)

    # Doing the discretization of the target by grouping into n bins where n is given by Sturge Rule
    n_bins = 1 + int(np.log2(df_train.shape[0])) 
    df_train["bins"] = pd.cut(df_train.target, bins = n_bins, labels = range(n_bins)).values


    # Create and fill the kfolds column
    df_train = df_train.sample(frac = 1, random_state = 42) # Shuffling
    df_train["kfolds"] = -1
    skf = StratifiedKFold(n_splits = k_folds)

    for i, (_, test_index) in enumerate(skf.split(df_train, df_train.bins)):
        df_train.loc[test_index, "kfolds"] = i

    # drop unnecessary columns
    df_train.drop("bins", axis = 1, inplace = True)

    # Write to csv
    output_name = f"train_{k_folds}_folds.csv"
    output_path = "../input_data"
    df_train.to_csv(os.path.join(output_path, output_name), index = False)


if __name__ == "__main__":
    generate_kfolds(10)