from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# Def eval metrics (mean square error, root mean square error, mean abs error). Note all are built in from scikit-learn

def eval_metrics(actual, forecast):

        mse = mean_squared_error(actual, forecast)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, forecast)
        r2 = r2_score(actual, forecast)

        results = {"RMSE": rmse, "MAE": mae, "R2": r2}

        return results

# Def Forecast functions (ARIMA, SARIMA, SARIMAX)

    # ARIMA Forecasting function no seasonality and no exogenous variables

def arima_forecast(ts_data, n_forecast, target_col= None):

        ts_targ = ts_data if isinstance(ts_data, pd.Series) else ts_data[target_col]

        n_forecast = int(n_forecast)

        arima_model = auto_arima(ts_targ, seasonal=False, stepwise=True, suppress_warnings=True)
        arima_pred, conf_int = arima_model.predict(n_periods=n_forecast, return_conf_int=True)

        return arima_model, arima_pred, conf_int

    # SARIMA Forecasting function with seasonality and no exogenous variables

def sarima_forecast(ts_data, n_forecast, target_col= None):

        ts_targ = ts_data if isinstance(ts_data, pd.Series) else ts_data[target_col]

        n_forecast = int(n_forecast)

        arima_model = auto_arima(ts_targ, seasonal=True, stepwise=True, suppress_warnings=True)
        arima_pred, conf_int = arima_model.predict(n_periods=n_forecast, return_conf_int=True)

        return arima_model, arima_pred, conf_int

    # SARIMAX Forecasting function with seasonality and exogenous variables

def sarimax_forecast(ts_data, n_forecast, target_col=None):

    if isinstance(ts_data, pd.Series):
        exog_cols = None

    else:
        exog_cols = ts_data.columns.drop(target_col).tolist()

    n_forecast = int(n_forecast)

    if exog_cols is None:
        ts_targ = ts_data
        arima_model = auto_arima(ts_targ, seasonal=True, m=12, error_action="ignore", suppress_warnings=True)
        arima_pred, conf_int = arima_model.predict(n_periods=n_forecast, return_conf_int=True)
    else:
        ts_targ = ts_data[target_col]
        ts_exog = ts_data[exog_cols]

        arima_model = auto_arima(ts_targ, X=ts_exog, seasonal=True, m=12, error_action="ignore", suppress_warnings=True)

        future_exog = pd.DataFrame({exog: auto_arima(ts_data[exog], seasonal=True, m=12, error_action="ignore", suppress_warnings=True).predict(n_periods=n_forecast) for exog in exog_cols}) # forecast future values of exogenous variables. 

        arima_pred, conf_int = arima_model.predict(n_periods=n_forecast, X=future_exog, return_conf_int=True)

    return arima_model, arima_pred, conf_int

    # SARIMAX with LAGGED exogenous variables

def sarimax_lagged_forecast(ts_data, n_forecast, target_col=None, lag=None):
 
    if isinstance(ts_data, pd.Series):
        raise ValueError("sarimax_lagged_forecast needs exogenous columns - pass a DataFrame and target_col.")
 
    exog_cols = ts_data.columns.drop(target_col).tolist()
    n_forecast = int(n_forecast)
    lag = n_forecast if lag is None else int(lag)
 
    if lag < n_forecast:
        raise ValueError("lag must be >= n_forecast, otherwise some future exog values would still be unknown.")
 
    ts_targ = ts_data[target_col]
    ts_exog = ts_data[exog_cols]
 
    n_obs = len(ts_data)
 
    # Training pairs: target at row i explained by exog at row i - lag (already observed).
    y_fit = ts_targ.iloc[lag:].to_numpy()
    X_fit = ts_exog.iloc[: n_obs - lag].to_numpy()
 
    arima_model = auto_arima(y_fit, X=X_fit, seasonal=True, m=12, error_action="ignore", suppress_warnings=True)
 
    # Future exog for the next n_forecast months = exog values from `lag` months ago,
    # which are the most recent rows already sitting in ts_exog - no forecasting needed.
    future_exog = ts_exog.iloc[n_obs - lag: n_obs - lag + n_forecast].to_numpy()
 
    arima_pred, conf_int = arima_model.predict(n_periods=n_forecast, X=future_exog, return_conf_int=True)
 
    return arima_model, arima_pred, conf_int


# Def FORECAST TEST function to evaluate the performance of the forecast functions using time series cross-validation

    # ARIMA Test

def arima_test(ts_data, n_splits, test_size, target_col=None, progress=True):

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    results = []

    print(f"Folds: {n_splits}")
    print(f"Test Size: {test_size}")

    folds = tqdm(tscv.split(ts_data), total=n_splits, desc="ARIMA", unit="fold", disable=not progress)

    for train_idx, test_idx in folds:
        ts_data_train = ts_data.iloc[train_idx]
        ts_data_test = ts_data.iloc[test_idx]

        _, arima_pred, _ = arima_forecast(ts_data_train, len(ts_data_test), target_col=target_col)

        actual = ts_data_test if isinstance(ts_data_test, pd.Series) else ts_data_test[target_col]

        fold_metrics = eval_metrics(actual, arima_pred)
        results.append(fold_metrics)
        folds.set_postfix(RMSE=f"{fold_metrics['RMSE']:.3f}")

    fold = pd.Series([*range(1, n_splits + 1)], name='Fold')
    results = pd.DataFrame(results)

    df = pd.concat([fold, results], axis=1)
    df = df.set_index('Fold')

    return df

    # SARIMA Test

def sarima_test(ts_data, n_splits, test_size, target_col=None, progress=True):

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    results = []

    print(f"Folds: {n_splits}")
    print(f"Test Size: {test_size}")

    folds = tqdm(tscv.split(ts_data), total=n_splits, desc="sarima", unit="fold", disable=not progress)

    for train_idx, test_idx in folds:
        ts_data_train = ts_data.iloc[train_idx]
        ts_data_test = ts_data.iloc[test_idx]

        _, arima_pred, _ = sarima_forecast(ts_data_train, len(ts_data_test), target_col=target_col)

        actual = ts_data_test if isinstance(ts_data, pd.Series) else ts_data_test[target_col]

        fold_metrics = eval_metrics(actual, arima_pred)
        results.append(fold_metrics)
        folds.set_postfix(RMSE=f"{fold_metrics['RMSE']:.3f}")

    fold = pd.Series([*range(1, n_splits + 1)], name='Fold')
    results = pd.DataFrame(results)

    df = pd.concat([fold, results], axis=1)
    df = df.set_index('Fold')

    return df


    # SARIMAX Test

def sarimax_test(ts_data, n_splits, test_size, target_col=None, progress=True):

    if isinstance(ts_data, pd.Series):
        exog_cols = None

    else:
        exog_cols = ts_data.columns.drop(target_col).tolist()

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    results = []

    print(f"Folds: {n_splits}")
    print(f"Test Size: {test_size}")

    folds = tqdm(tscv.split(ts_data), total=n_splits, desc="sarimax", unit="fold", disable=not progress)

    for train_idx, test_idx in folds:
        ts_data_train = ts_data.iloc[train_idx]
        ts_data_test = ts_data.iloc[test_idx]

        _, arima_pred, _ = sarimax_forecast(ts_data_train, len(ts_data_test), target_col=target_col)

        actual = ts_data_test if exog_cols is None else ts_data_test[target_col]

        fold_metrics = eval_metrics(actual, arima_pred)
        results.append(fold_metrics)
        folds.set_postfix(RMSE=f"{fold_metrics['RMSE']:.3f}")

    fold = pd.Series([*range(1, n_splits + 1)], name='Fold')
    results = pd.DataFrame(results)

    df = pd.concat([fold, results], axis=1)
    df = df.set_index('Fold')

    return df



    # SARIMAX with LAGGED exogenous variables Test

def sarimax_lagged_test(ts_data, n_splits, test_size, target_col=None, lag=None, progress=True):
 
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    results = []
 
    print(f"Folds: {n_splits}")
    print(f"Test Size: {test_size}")
 
    folds = tqdm(tscv.split(ts_data), total=n_splits, desc="sarimax_lagged", unit="fold", disable=not progress)
 
    for train_idx, test_idx in folds:
        ts_data_train = ts_data.iloc[train_idx]
        ts_data_test = ts_data.iloc[test_idx]
 
        _, arima_pred, _ = sarimax_lagged_forecast(ts_data_train, len(ts_data_test), target_col=target_col, lag=lag)
 
        actual = ts_data_test[target_col]
 
        fold_metrics = eval_metrics(actual, arima_pred)
        results.append(fold_metrics)
        folds.set_postfix(RMSE=f"{fold_metrics['RMSE']:.3f}")
 
    fold = pd.Series([*range(1, n_splits + 1)], name='Fold')
    results = pd.DataFrame(results)
 
    df = pd.concat([fold, results], axis=1)
    df = df.set_index('Fold')
 
    return df   


## SAMPLE FITTING AND TESTING

def sarimax_fitted(model, ts_data, target_col, exog_lag=0):
    """
    Reconstruct the fitted model's in-sample predictions, aligned to real dates.

    model      : the fitted pmdarima model (e.g. sarimax_model, lagged_model)
    ts_data    : the same DataFrame that was passed in when fitting `model`
    target_col : the target column name
    exog_lag   : 0 for a model from sarimax_forecast (contemporaneous exog).
                 the `lag` value used for sarimax_lagged_forecast /
                 sarimax_lagged_extended_forecast (lagged exog).
    """
    exog_cols = ts_data.columns.drop(target_col).tolist()
    ts_targ = ts_data[target_col]
    ts_exog = ts_data[exog_cols]
    n_obs = len(ts_data)

    if exog_lag == 0:
        X = ts_exog.to_numpy()
        idx = ts_targ.index
    else:
        X = ts_exog.iloc[: n_obs - exog_lag].to_numpy()
        idx = ts_targ.index[exog_lag:]

    fitted = model.predict_in_sample(X=X)
    return pd.Series(fitted, index=idx, name="Fitted")


def sarimax_fitted_metrics(model, ts_data, target_col, exog_lag=0):
    """RMSE / MAE / R2 of the in-sample fit vs. the real historical values."""
    fitted = sarimax_fitted(model, ts_data, target_col, exog_lag)
    actual = ts_data[target_col].loc[fitted.index]
    return eval_metrics(actual, fitted)


def plot_sarimax_fitted(model, ts_data, target_col, exog_lag=0, start=None, title=None, figsize=(14, 5)):
    """Plot actual vs. in-sample fitted values for a fitted SARIMAX/lagged-SARIMAX model."""
    fitted = sarimax_fitted(model, ts_data, target_col, exog_lag)
    actual = ts_data[target_col]
    m = eval_metrics(actual.loc[fitted.index], fitted)

    plt.figure(figsize=figsize)
    plt.plot(actual.loc[start:], color="black", label="Actual")
    plt.plot(fitted.loc[start:], linestyle="--", color="tab:red", label="Fitted (in-sample)")
    plt.grid(True)
    plt.xlabel("Date")
    plt.ylabel("Monthly YoY Inflation Rate (%)")
    plt.title(title or f"In-sample fit: RMSE {m['RMSE']:.3f}, MAE {m['MAE']:.3f}, R2 {m['R2']:.3f}")
    plt.legend()
    plt.show()
    return m