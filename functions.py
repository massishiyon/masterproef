import numpy as np
import pandas as pd


def normalize(feature, train_ind):
    Var_train=feature.loc[train_ind]
    Var_train_mean = np.mean(Var_train)
    Var_train_std = np.std(Var_train)

    #Standardize variable
    Var_preprocessed = (feature-Var_train_mean)/Var_train_std

    #Deal with outliers
    j=0
    for i in Var_preprocessed:
        if (i>3):
            Var_preprocessed.iloc[j] = 3
        elif (i<-3):
            Var_preprocessed.iloc[j] = -3
        j+=1
    return Var_preprocessed

def generate_dummies(feature, var_name):
    Var_preprocessed = pd.get_dummies(feature, prefix=str(var_name))
    # leave the first category out, so it becomes the reference category
    Var_preprocessed = Var_preprocessed.iloc[:, 1:]
    return Var_preprocessed