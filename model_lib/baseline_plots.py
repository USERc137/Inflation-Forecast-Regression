import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


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
