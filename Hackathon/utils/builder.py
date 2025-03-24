import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor


def build_linear_regression(context, target):
    model = LinearRegression()
    model.fit(context, target)
    return model


def build_random_forest(context, target):
    model = RandomForestRegressor()
    model.fit(context, target)
    return model


def build_hgbr(context, target):
    model = HistGradientBoostingRegressor()
    model.fit(context, target)
    return model


def build_xgbr(context, target):
    model = XGBRegressor()
    model.fit(context, target)
    return model
