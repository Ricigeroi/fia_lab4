import numpy as np
import pandas as pd
from matplotlib.pyplot import figure
from scipy.cluster.hierarchy import average
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder


# loading data from .csv into pandas dataframe
dataset = pd.read_csv('datasets/data.csv')


# looking for empty entries
print(dataset.info())


def preprocess(df):
    # replace 'Not Provided' to NaN
    df['BasePay'] = df['BasePay'].replace('Not Provided', np.nan)
    df['Benefits'] = df['Benefits'].replace('Not Provided', np.nan)
    df['OvertimePay'] = df['OvertimePay'].replace('Not Provided', np.nan)
    df['OtherPay'] = df['OtherPay'].replace('Not Provided', np.nan)

    # convert column to numeric
    df['BasePay'] = pd.to_numeric(df['BasePay'], errors='coerce')
    df['Benefits'] = pd.to_numeric(df['Benefits'], errors='coerce')
    df['OvertimePay'] = pd.to_numeric(df['Benefits'], errors='coerce')
    df['OtherPay'] = pd.to_numeric(df['Benefits'], errors='coerce')

    # filling empty benefits entries with median value
    median_BasePay = df['BasePay'].median()
    df['BasePay'] = df['BasePay'].fillna(median_BasePay)

    # filling empty OvertimePay entries with median value
    median_OvertimePay = df['OvertimePay'].median()
    df['OvertimePay'] = df['OvertimePay'].fillna(median_OvertimePay)

    # filling empty OtherPay entries with median value
    median_OtherPay = df['OtherPay'].median()
    df['OtherPay'] = df['OtherPay'].fillna(median_OtherPay)

    # filling empty benefits entries with mean value
    average_Benefits = df['Benefits'].mean()
    df['Benefits'] = df['Benefits'].fillna(average_Benefits)

    # replace NaN to 'Not Provided' for encoding purposes
    df['Status'] = df['Status'].fillna("Not Provided")

    # one-hot encoding Status column
    status_encoder = OneHotEncoder(sparse_output=False)
    status_encoded = status_encoder.fit_transform(df[['Status']])
    status_encoded_df = pd.DataFrame(status_encoded, columns=status_encoder.get_feature_names_out())
    df.reset_index(drop=True, inplace=True)
    status_encoded_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, status_encoded_df], axis=1).drop(['Status'], axis=1)

    # label encoding JobTitle
    jobtitle_encoder = LabelEncoder()
    df['JobTitle'] = jobtitle_encoder.fit_transform(df['JobTitle'])

    # scaling
    scaler = MinMaxScaler()
    df['BasePay'] = scaler.fit_transform(df[['BasePay']])
    df['OvertimePay'] = scaler.fit_transform(df[['OvertimePay']])
    df['OtherPay'] = scaler.fit_transform(df[['OtherPay']])
    df['Benefits'] = scaler.fit_transform(df[['Benefits']])
    df['TotalPay'] = scaler.fit_transform(df[['TotalPay']])
    df['TotalPayBenefits'] = scaler.fit_transform(df[['TotalPayBenefits']])
    df['Year'] = scaler.fit_transform(df[['Year']])

    return df.drop(columns=['Id', 'Notes','EmployeeName', 'Agency'])


dataset = preprocess(dataset)

