import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def gather_data(first_year, last_year):
    years = []
    for i in range(first_year, last_year+1):
        months = []
        for j in range(1,13):
            months.append(pd.read_csv(f"data/PRICE_AND_DEMAND_{i}{j:02}_VIC1.csv"))
        years.append(pd.concat(months))
    data = pd.concat(years)
    data["SETTLEMENTDATE"] = pd.to_datetime(data["SETTLEMENTDATE"])
    data["hour"] = data["SETTLEMENTDATE"].dt.hour       # hour of day
    data["minute"] = data["SETTLEMENTDATE"].dt.minute + 60*data["hour"]
    data["DoY"] = data["SETTLEMENTDATE"].dt.dayofyear   # day of year


    # take sin and cosine. maps time to a point on the unit circle essentially
    data["minute_sin"] = data["minute"].apply(lambda x: np.sin(x*2*np.pi/1440))  # sine of hour to capture daily swings
    data["minute_cos"] = data["minute"].apply(lambda x: np.cos(x*2*np.pi/1440))  # cosine of hour ""
    data["DoY_sin"] = data["DoY"].apply(lambda x: np.sin(x*2*np.pi/365))    # sine of DoY to capture seasonality
    data["DoY_cos"] = data["DoY"].apply(lambda x: np.cos(x*2*np.pi/365))    # cosine of DoY ""

    # drop unnecessary data
    data = data.drop(columns=["REGION", "PERIODTYPE", "hour"])
    return data.sort_values("SETTLEMENTDATE", ignore_index=True)




def add_horizons(df, horizons, frac=1):
    df = df.sort_values("SETTLEMENTDATE").reset_index(drop=True)
    dfs = []
    
    for h in horizons:
        temp = df.copy()
        
        temp["target"] = temp["RRP"].shift(-h)
        temp["h"] = h
        
        temp["SETTLEMENTDATE_future"] = temp["SETTLEMENTDATE"] + pd.to_timedelta(5 * h, unit="m")
        
        hour_future = temp["SETTLEMENTDATE_future"].dt.hour
        minute_future = temp["SETTLEMENTDATE_future"].dt.minute + 60 * hour_future
        temp["minute_future"] = minute_future
        temp["minute_sin_future"] = np.sin(minute_future * 2 * np.pi / 1440)
        temp["minute_cos_future"] = np.cos(minute_future * 2 * np.pi / 1440)

        doy_future = temp["SETTLEMENTDATE_future"].dt.dayofyear
        temp["DoY_future"] = doy_future
        temp["DoY_sin_future"] = np.sin(doy_future * 2 * np.pi / 365)
        temp["DoY_cos_future"] = np.cos(doy_future * 2 * np.pi / 365)
        # reduce rows if necessary
        dfs.append(temp.sample(frac=frac, random_state=0))
    
    df_multi = pd.concat(dfs, ignore_index=True)
    df_multi = df_multi.dropna(subset=["target"])

    # reduce numerical data size
    float_cols = df_multi.select_dtypes(include="float64").columns
    int_cols = df_multi.select_dtypes(include="int64").columns

    df_multi[float_cols] = df_multi[float_cols].astype("float32")
    df_multi[int_cols] = df_multi[int_cols].astype("int32")





    
    return df_multi


def prepare_data(first_year, last_year, horizons, split_date, frac=1.):
    df = gather_data(first_year, last_year)
    
    # split before adding horizons to prevent data leakage
    train_df = df[df["SETTLEMENTDATE"] <= split_date]
    test_df = df[df["SETTLEMENTDATE"] > split_date]
    
    train = add_horizons(train_df, horizons, frac)
    test = add_horizons(test_df, horizons, frac=1.)

    return train, test, test_df

def train_model(train, test):
    drop_cols = ["target", "SETTLEMENTDATE", "SETTLEMENTDATE_future"]
    features = [c for c in train.columns if c not in drop_cols]

    X_train, y_train = train[features], train["target"]
    X_test,  y_test  = test[features],  test["target"]
    
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric="mae",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )


    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

    return model, preds, y_test

def make_inference_df(settlement_date, demand, rrp, horizons):
    h = np.asarray(list(horizons))  # ensure numpy array

    # future times (vectorized)
    future_time = settlement_date + pd.to_timedelta(5 * h, unit="m")

    # current time features (scalar → reused)
    minute = settlement_date.hour * 60 + settlement_date.minute
    doy = settlement_date.dayofyear

    # future features (vectorized)
    minute_future = future_time.hour * 60 + future_time.minute
    doy_future = future_time.dayofyear

    # build dataframe in one shot
    df = pd.DataFrame({
        "TOTALDEMAND": demand,
        "RRP": rrp,
        "minute": minute,
        "DoY": doy,
        "minute_sin": np.sin(minute * 2 * np.pi / 1440),
        "minute_cos": np.cos(minute * 2 * np.pi / 1440),
        "DoY_sin": np.sin(doy * 2 * np.pi / 365),
        "DoY_cos": np.cos(doy * 2 * np.pi / 365),

        "h": h,

        "minute_future": minute_future,
        "minute_sin_future": np.sin(minute_future * 2 * np.pi / 1440),
        "minute_cos_future": np.cos(minute_future * 2 * np.pi / 1440),

        "DoY_future": doy_future,
        "DoY_sin_future": np.sin(doy_future * 2 * np.pi / 365),
        "DoY_cos_future": np.cos(doy_future * 2 * np.pi / 365),
    })

    return df


def forecast(settlement_date, demand, rrp, horizons, model):
    df = make_inference_df(settlement_date, demand, rrp, horizons)

    return model.predict(df)