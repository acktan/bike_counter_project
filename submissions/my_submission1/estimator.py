import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import arrow
import dateutil.parser
import math
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

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



def get_estimator():
    date_encoder = FunctionTransformer(_cycl_encode)
    date_cols = ['year_sin', 'year_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'weekday_sin', 'weekday_cos', 'hour_sin', 'hour_cos']

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]
    numeric_cols = ['t', 'ff', 'u']

    preprocessor = ColumnTransformer(
        [
            ("date", "passthrough", date_cols),
            ("cat", categorical_encoder, categorical_cols)
        ]
    )
    regressor = HistGradientBoostingRegressor(random_state=0)

    pipe = make_pipeline(date_encoder, preprocessor, regressor)

    return pipe
