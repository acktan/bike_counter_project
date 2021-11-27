import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor

__file__ = Path('submissions') /  'external_data' /  'estimator.py'

def _cycl_encode(X):
    X = X.copy()
    year_norm = 2 * math.pi * X['date'].dt.year / X['date'].dt.year.max()
    month_norm = 2 * math.pi * X['date'].dt.month / X['date'].dt.month.max()
    day_norm = 2 * math.pi * X['date'].dt.day / X['date'].dt.day.max()
    weekday_norm = 2 * math.pi * X['date'].dt.weekday / X['date'].dt.weekday.max()
    hour_norm = 2 * math.pi * X['date'].dt.hour / X['date'].dt.hour.max()
    X.loc[:, 'year_sin'] = np.sin(year_norm)
    X.loc[:, 'year_cos'] = np.cos(year_norm)
    X.loc[:, 'month_sin'] = np.sin(month_norm)
    X.loc[:, 'month_cos'] = np.cos(month_norm)
    X.loc[:, 'day_sin'] = np.sin(day_norm)
    X.loc[:, 'day_cos'] = np.cos(day_norm)
    X.loc[:, 'weekday_sin'] = np.sin(weekday_norm)
    X.loc[:, 'weekday_cos'] = np.cos(weekday_norm)
    X.loc[:, 'hour_sin'] = np.sin(hour_norm)
    X.loc[:, 'hour_cos'] = np.cos(hour_norm)

    return X.drop(columns=["date"]) 


def _merge_external_data(X):
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X['orig_index'] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values('date'), df_ext[['date', 't', 'ff', 'u', 'brent', 'vacances_scol', 'bank_hol']].sort_values('date'), on='date')
    # Sort back to the original order
    X = X.sort_values('orig_index')
    del X['orig_index']
    return X

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, 'year'] = X['date'].dt.year
    X.loc[:, 'month'] = X['date'].dt.month
    X.loc[:, 'day'] = X['date'].dt.day
    X.loc[:, 'weekday'] = X['date'].dt.weekday
    X.loc[:, 'hour'] = X['date'].dt.hour

    return X.drop(columns=["date"]) 

    # Finally we can drop the original columns from the dataframe

def get_estimator():
    #date_encoder = FunctionTransformer(_encode_dates)
    #date_cols = ['year', 'month', 'day', 'weekday', 'hour']

    cycl_encoder = FunctionTransformer(_cycl_encode)
    cycl_cols = ['year_sin', 'year_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos', 'hour_sin', 'hour_cos']

    categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    categorical_cols = ["site_name", "counter_name"]

    numeric_cols = ['t', 'ff', 'u', 'brent', 'bank_hol', 'vacances_scol']

    preprocessor = ColumnTransformer(
        [
            ('date', 'passthrough', cycl_cols),
            ('cat', categorical_encoder, categorical_cols),
            ('numeric', 'passthrough', numeric_cols)
        ]
    )
    regressor = HistGradientBoostingRegressor(random_state=0)

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False), cycl_encoder, preprocessor, regressor)

    return pipe
