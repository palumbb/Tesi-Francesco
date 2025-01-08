import numpy as np
import pandas as pd 
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype


def dirty(seed, df, features, method, dirty_percentage): # name_class sarebbe il nome della colonna che contiene le class labels. Così li non mettono i missing values
    if method == "uniform":
        return uniform(seed, df, features, dirty_percentage)

# IMPUTE WITH O, MISSING
def uniform(seed, df, features, dirty_percentage):
    np.random.seed(seed)

    df_dirt = df.copy()
    comp = [dirty_percentage,1-dirty_percentage]
    df_dirt = check_datatypes(df_dirt)

    for col in df_dirt.columns:
        if col in features:
            rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)
            if is_bool_dtype(df_dirt[col]) | is_object_dtype(df_dirt[col]):
                df_dirt.loc[rand == True,col] = "missing"
            elif is_numeric_dtype(df_dirt[col]):
                df_dirt.loc[rand == True,col] = 0
            elif df_dirt[col].dtype == "float64":
                df_dirt.loc[rand == True,col] = 0.0
            elif col == "int64":
                df_dirt.loc[rand == True,col] = 0

        # print("saved {}-completeness{}%".format(name, round((1-p)*100)))
    return df_dirt 

# IMPUTE WITH NAN
def uniform_nan(seed, df, features, dirty_percentage):
    np.random.seed(seed)

    df_dirt = df.copy()
    comp = [dirty_percentage,1-dirty_percentage]
    df_dirt = check_datatypes(df_dirt)

    for col in df_dirt.columns:
                if col in features:
                    rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)
                    df_dirt.loc[rand == True,col]=np.nan

    return df_dirt

def check_datatypes(df):
    for col in df.columns:
        if (df[col].dtype == "bool"):
            df[col] = df[col].astype('string')
            df[col] = df[col].astype('object')
    return  df

def impute_missing_column(df, method):
    np.random.seed(0)
    if method == "impute_standard":
        imputator = impute_standard()
        imputated_df = imputator.fit(df)
    elif method == "impute_mean":
        imputator = impute_mean()
        imputated_df = imputator.fit_mode(df)
    return imputated_df


class impute_mean:
    def __init__(self):
        self.name = 'Mean'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(df[col].mean())
        return df

    def fit_mode(self, df):
        for col in df.columns:
            if is_bool_dtype(df[col]) | is_object_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mode()[0])
                #df[col] = df[col].replace("missing", df[df[col]!="missing"][col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())
                #df[col] = df[col].replace(0, df[df[col]!=0][col].mean())
                #df[col] = df[col].replace(0.0, df[df[col]!=0][col].mean())
        return df
    
class impute_standard:
    def __init__(self):
        self.name = 'Standard'

    def fit(self, df):
        for col in df.columns:
            if (df[col].dtype != "object"):
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna("missing")
        return df
