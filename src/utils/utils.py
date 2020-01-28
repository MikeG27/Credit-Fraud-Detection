import pandas as pd
import numpy as np


def reduce_memory(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe was {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def df_basic_info(df):
    print("Basic summary : \n")
    print("Features : ",df.shape[1])
    print("\nPCA Features :", len(df.columns[1:29]),"--", df.iloc[:,1:29].dtypes[0])
    print("Other Features : Time --",df.Time.dtype,"Amount --",df.Amount.dtype)
    print("Labels :",df.columns[-1], df.Class.unique(),"--", df.Class.dtype)
    print("\nRows : ",df.shape[0])
    print("\nClass normal : ", len(df[df.Class == 0]))
    print("Fraud : ",len(df[df.Class == 1]))
    print("Fraud cases percentage : ",round(len(df[df.Class == 1])/len(df[df.Class == 0])*100,4),"%")
    print("\nMissing values : ",df.isnull().sum().sum())


def print_best_corelations(df):
    df_corr = df.corr().unstack().sort_values().drop_duplicates()
    print("Top five positive corelation : \n")
    print(df_corr.tail().sort_values(ascending=False))

    print("\nTop five negative corelation : \n")
    print(df_corr.head())


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y