import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.model_selection import TimeSeriesSplit # used for time-series data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # standard evaluation metrics
from pandas.tseries.frequencies import to_offset # for parsing frequency data

##---------------------------------------------------------------------------


def naive_forecast(data, target_col, months_ahead):
    if isinstance(data, pd.Series):
        data = data.to_frame(name = target_col if data.name is None else data.name)

    data = data[target_col] # to extract the column being forecast

    freq = data.index.freq or to_offset(pd.infer_freq(data.index)) # figures out the date frequency
    future_indices = pd. date_range(start= data.index[-1] + freq, periods = months_ahead, freq= freq)

    future_values = pd.Series([data.iloc[-1] for _ in range(months_ahead)], index= future_indices) 
    # the future value (inflation forecast) is the last value repeated by months_head times. 

    return future_values

def seasonal_naive_forecast(data, target_col, months_ahead):

    if isinstance(data, pd.Series):
        data = data.to_frame(name= target_col if data.name is None else data.name)

    data = data[target_col]
    if len(data) < 12:
        raise ValueError("At least 12 observations are required form seasonal naive forecast")
        # needs at least one full year of history. 

    # building the future dates
    freq = data.index.freq or to_offset(pd.infer_freq(data.index))
    future_indices = pd. date_range(start= data.index[-1] + freq, periods = months_ahead, freq= freq)

    # Seasonality 
    future_values = pd.Series([data.iloc[-12 + (i % 12)] for i in range(months_ahead)], index = future_indices)

    return future_values

# defining the evaluation metric (mean square error, root mean square error, mean abs error). Note all are built in from scikit-learn

def eval_metrics(actual, forecast):

    mse = mean_squared_error(actual, forecast)   # average of squared errors
    rmse = np.sqrt(mse)                          # back to original units (% points)
    mae = mean_absolute_error(actual, forecast)  # average absolute error
    r2 = r2_score(actual, forecast)              # share of variance explained

    results = {"RMSE": rmse, "MAE": mae, "R2": r2}

    return results


# "actual" is indexed by test dates and "forecast" by future dates. sklearn does not see this, it compares by position.
# In this case, this works as long as both have the same length and order. 


def forecast_test(ts_data, forecast_fn, n_splits, test_size, target_col=None, progress=True):

    # Allow a plain Series with no target_col (the old default would have crashed)
    if isinstance(ts_data, pd.Series):
        target_col = target_col or ts_data.name or "value"
        ts_data = ts_data.to_frame(name=target_col)
    elif target_col is None:
        raise ValueError("target_col is required when ts_data is a DataFrame.")

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    results = []

    print(f"Model: {forecast_fn.__name__}")
    print(f"Folds: {n_splits}")
    print(f"Test Size: {test_size}")

    # progress bars -------
    folds = tqdm(
        tscv.split(ts_data),
        total=n_splits,
        desc=forecast_fn.__name__,
        unit="fold",
        disable=not progress,
    )
    for train_idx, test_idx in folds:
        train = ts_data.iloc[train_idx]
        test = ts_data.iloc[test_idx]

        forecast = forecast_fn(train, target_col, test_size)
        actual = test[target_col]

        results.append(eval_metrics(actual, forecast))

    df = pd.DataFrame(results)
    df.index = pd.RangeIndex(1, len(df) + 1, name="Fold")
    return df

##################


def naive_test(ts_data, n_splits, test_size, target_col=None, progress=True):
    return forecast_test(ts_data, naive_forecast, n_splits, test_size, target_col, progress)


def seasonal_naive_test(ts_data, n_splits, test_size, target_col=None, progress=True):
    return forecast_test(ts_data, seasonal_naive_forecast, n_splits, test_size, target_col, progress)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    idx = pd.date_range("1990-01-01", periods=300, freq="MS")
    y = 4 + 0.8 * np.sin(2 * np.pi * idx.month / 12) + rng.normal(0, 0.2, 300)
    df = pd.DataFrame({"InflationRate": y}, index=idx)

    print(naive_test(df, n_splits=5, test_size=12, target_col="InflationRate").round(3), "\n")
    print(seasonal_naive_test(df, n_splits=5, test_size=12, target_col="InflationRate").round(3), "\n")
    # Series input, no target_col needed
    print(forecast_test(df["InflationRate"], naive_forecast, 3, 12).round(3))

# Plot the results

def _to_series(data, target_col=None):

    if isinstance(data, pd.DataFrame):
        if target_col is None:
            raise ValueError("target_col is required for a DataFrame.")
        return data[target_col]
    return data

# --------------------------------------------------------------------------
# 1. In-sample "fitted" values
# --------------------------------------------------------------------------
# Naive and seasonal naive have no parameters to train, so their "fit" is the
# one-step-ahead prediction they would have made at each point in the past:
#   naive:          y_hat[t] = y[t-1]
#   seasonal naive: y_hat[t] = y[t-12]
def naive_fitted(y):
    return y.shift(1).rename("Naive")
 
 
def seasonal_naive_fitted(y, season_length=12):
    return y.shift(season_length).rename("Seasonal Naive")
 
 
def fitted_metrics(y, fitted):
    mask = y.notna() & fitted.notna()
    err = y[mask] - fitted[mask]
    return {"RMSE": float(np.sqrt((err ** 2).mean())), "MAE": float(err.abs().mean())}
 
 
def plot_fitted(data, target_col=None, start=None, season_length=12, figsize=(16, 8)):
    """Actual vs. fitted values for both baselines (one panel each) + metrics table."""
    y = _to_series(data, target_col)
    fits = [naive_fitted(y), seasonal_naive_fitted(y, season_length)]
 
    fig, axes = plt.subplots(len(fits), 1, figsize=figsize, sharex=True)
    rows = {}
    for ax, fit in zip(axes, fits):
        m = fitted_metrics(y, fit)          # scored on the full sample
        rows[fit.name] = m
        ax.plot(y.loc[start:], color="black", label="Actual")
        ax.plot(fit.loc[start:], linestyle="--", label=f"{fit.name} (fitted)")
        ax.set_title(f"{fit.name}: RMSE {m['RMSE']:.3f}, MAE {m['MAE']:.3f}")
        ax.set_ylabel("YoY Inflation Rate (%)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(rows).T
 
 
# --------------------------------------------------------------------------
# 2. Out-of-sample backtest from the CV folds
# --------------------------------------------------------------------------
def plot_cv_forecasts(data, forecast_fn, n_splits, test_size, target_col=None,
                      start=None, title=None, figsize=(16, 5)):
    """
    Draw the multi-step forecast from each CV fold over the actual series.
    forecast_fn has the signature forecast_fn(train, target_col, months_ahead).
    """
    if isinstance(data, pd.Series):
        target_col = target_col or data.name or "value"
        data = data.to_frame(name=target_col)
 
    y = data[target_col]
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
 
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y.loc[start:], color="black", label="Actual")
    for k, (tr, te) in enumerate(tscv.split(data), start=1):
        fc = forecast_fn(data.iloc[tr], target_col, test_size)
        ax.axvspan(y.index[te[0]], y.index[te[-1]], alpha=0.07, color="tab:blue")
        ax.plot(fc, linewidth=2, label=f"Fold {k}")
    ax.set_title(title or f"{forecast_fn.__name__}: forecast per CV fold vs. actual")
    ax.set_ylabel("YoY Inflation Rate (%)")
    ax.set_xlabel("Date")
    ax.grid(alpha=0.3)
    ax.legend(ncol=n_splits + 1, loc="upper left")
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    from baseline_tests import naive_forecast, seasonal_naive_forecast
 
    rng = np.random.default_rng(0)
    idx = pd.date_range("1990-01-01", periods=300, freq="MS")
    y = 4 + 0.8 * np.sin(2 * np.pi * idx.month / 12) + rng.normal(0, 0.2, 300)
    df = pd.DataFrame({"InflationRate": y}, index=idx)
 
    print(plot_fitted(df, "InflationRate", start="2010"))
    plot_cv_forecasts(df, naive_forecast, 5, 12, "InflationRate", start="2010")
    plot_cv_forecasts(df, seasonal_naive_forecast, 5, 12, "InflationRate", start="2010")