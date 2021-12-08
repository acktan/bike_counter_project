from numpy.lib.shape_base import apply_along_axis
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem

def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )
  
def _encode(X):
    #cyclical encoding of dates
    X = X.copy()
    year_norm = 2 * np.pi * X['date'].dt.year / X['date'].dt.year.max()
    month_norm = 2 * np.pi * X['date'].dt.month / X['date'].dt.month.max()
    day_norm = 2 * np.pi * X['date'].dt.day / X['date'].dt.day.max()
    weekday_norm = 2 * np.pi * X['date'].dt.weekday / X['date'].dt.weekday.max()
    hour_norm = 2 * np.pi * X['date'].dt.hour / X['date'].dt.hour.max()
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
    #encode dates
    X.loc[:, 'year'] = X['date'].dt.year
    X.loc[:, 'month'] = X['date'].dt.month
    X.loc[:, 'day'] = X['date'].dt.day
    X.loc[:, 'weekday'] = X['date'].dt.weekday
    X.loc[:, 'hour'] = X['date'].dt.hour
    X.loc[:, 'workday'] = np.where(X['weekday'].isin([0,4]), 1,0)
    return X.drop(columns=["date"]) 
 
def _merge_external_data(X):
    file_path = Path(__file__).parent / 'external_data.csv'
    df_ext = pd.read_csv(file_path, parse_dates=['date'])
    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X['orig_index'] = np.arange(X.shape[0])
    X = pd.merge_asof(X.sort_values('date'), df_ext[['date', 't', 'ff', 'u', 'brent', 'holidays', 'curfew', 'rush hour', 'Taux', 'workplace', 'transit']].sort_values('date'), on='date')
    # Sort back to the original order
    X = X.sort_values('orig_index')
    del X['orig_index']
    return X

def get_estimator():
    #columns
    date_encoder = FunctionTransformer(_encode)
    #cycl_cols = ['month_sin', 'month_cos','day_sin', 'day_cos', 'weekday_sin', 'weekday_cos', 'hour_sin', 'hour_cos']
    date_cols = ['year', 'day']

    #FunctionTransformer(imputer)

    categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    categorical_cols = ["site_name", "counter_name"]
    binary_cols =  ['curfew', 'rush hour']
    numeric_cols = ['Taux', 't', 'brent', 'ff', 'u', 'transit']
 
    #preprocessor
        
    preprocessor = ColumnTransformer(
        [
            ('date', 'passthrough', date_cols),
            #('cycl', 'passthrough', cycl_cols),
            ("cyclic_month", periodic_spline_transformer(12, n_splines=6), ["month"]),
            ("cyclic_weekday", periodic_spline_transformer(7, n_splines=3), ["weekday"]),
            ("cyclic_hour", periodic_spline_transformer(24, n_splines=12), ["hour"]),
            ('holiday', 'passthrough', binary_cols),  
            ('cat', categorical_encoder, categorical_cols),
            ('numeric', 'passthrough', numeric_cols)
        ]
    )
    hour_workday_interaction = make_pipeline(
        ColumnTransformer(
            [
            ("cyclic_hour", periodic_spline_transformer(24, n_splines=12), ["hour"]),
            ("weekday", FunctionTransformer(lambda x: x == "True"), ["workday"]),
            ]
        ),
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
    )

    regressor = lgb.LGBMRegressor(n_estimators=275, num_leaves=150, random_state=0)

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder, 
        FeatureUnion(
            [('without', preprocessor),
             ('with', hour_workday_interaction)
            ]
        ), 
        regressor)

    return pipe