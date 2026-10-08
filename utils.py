import os
import numpy as np
import math
import scipy as sc
import pandas as pd
#import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
#from plotly.subplots import make_subplots
pd.options.plotting.backend = "plotly"


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _smoothing_input(x, h, params, coefficients):
    _positive_int(h, 'h')
    values = np.asarray(x, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError('x must be a non-empty one-dimensional series')
    if np.isinf(values).any():
        raise ValueError('x must not contain infinity')
    for name in coefficients:
        value = params[name]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be finite and between 0 and 1')
    return values


def _adapted_weight(weight, age, period):
    # Start with a larger weight, then approach the requested coefficient.
    return 1 - (1 - weight) * min(age / period, 1)


def InitExponentialSmoothing(x, h, Params):
    """SES with an initial adaptation period; output[t+h] uses x[:t+1]."""
    values = _smoothing_input(x, h, Params, ['alpha'])
    period = _positive_int(Params['AdaptationPeriod'], 'AdaptationPeriod')
    forecast = [np.nan] * (len(values) + h)
    level, first = np.nan, None
    for t, value in enumerate(values):
        if not np.isnan(value):
            if first is None:
                level, first = value, t
            weight = _adapted_weight(Params['alpha'], t - first + 1, period)
            level = weight * value + (1 - weight) * level
        forecast[t + h] = level
    return forecast


def SimpleExponentialSmoothing(x, h=1, params=None, freq=None):
    """SES for a dated Series, preserving its index and extending it by h ticks."""
    if params is None:
        params = {'alpha': 0.1}
    values = _smoothing_input(x, h, params, ['alpha'])
    if not isinstance(x, pd.Series) or not isinstance(x.index, pd.DatetimeIndex):
        raise ValueError('x must be a pandas Series with a DatetimeIndex')
    if not x.index.is_unique or not x.index.is_monotonic_increasing:
        raise ValueError('x must have a unique, increasing index')
    if freq is None:
        freq = x.index.freqstr
        if freq is None and len(x) >= 3:
            freq = pd.infer_freq(x.index)
    if freq is None:
        raise ValueError('Provide freq for a short or irregular time series')
    future = pd.date_range(x.index[-1], periods=h + 1, freq=freq)[1:]
    forecast = pd.Series(np.nan, index=x.index.append(future), name='fcst ' + str(x.name), dtype=float)
    level = np.nan
    for t, value in enumerate(values):
        if not np.isnan(value):
            if np.isnan(level):
                level = value
            level = params['alpha'] * value + (1 - params['alpha']) * level
        forecast.iloc[t + h] = level
    return forecast


def HoltExponentialSmoothing(x, h, Params):
    """Causal Holt forecast with adaptive initial weights.

    The first observation initializes the level with zero trend. The second
    observed value initializes the slope (per time tick). Missing observations
    propagate the level along the current trend without updating the weights.
    """
    values = _smoothing_input(x, h, Params, ['alpha', 'beta'])
    period = _positive_int(Params['AdaptationPeriod'], 'AdaptationPeriod')
    forecast = [np.nan] * (len(values) + h)
    level, trend, first, have_trend = np.nan, 0.0, None, False
    for t, value in enumerate(values):
        if first is None:
            if not np.isnan(value):
                level, first, first_value = value, t, value
        elif np.isnan(value):
            level += trend
        elif not have_trend:
            trend = (value - first_value) / (t - first)
            level, have_trend = value, True
        else:
            alpha = _adapted_weight(Params['alpha'], t - first + 1, period)
            beta = _adapted_weight(Params['beta'], t - first + 1, period)
            previous_level = level
            level = alpha * value + (1 - alpha) * (level + trend)
            trend = beta * (level - previous_level) + (1 - beta) * trend
        forecast[t + h] = level + h * trend
    return forecast


def _seasonal_smoothing(x, h, params, with_trend):
    coefficients = ['alpha', 'gamma'] + (['beta'] if with_trend else [])
    values = _smoothing_input(x, h, params, coefficients)
    p = _positive_int(params['seasonality_period'], 'seasonality_period')
    forecast = [np.nan] * (len(values) + h)
    # Initialize only once complete seasons have actually been observed.
    warmup = (2 if with_trend else 1) * p
    start = next((i for i in range(len(values) - warmup + 1)
                  if np.isfinite(values[i:i + warmup]).all()), None)
    if start is None:
        return forecast
    end = start + warmup - 1
    initial = values[start:end + 1]
    trend = (initial[p:].mean() - initial[:p].mean()) / p if with_trend else 0.0
    level = initial.mean() + trend * (warmup - 1) / 2
    detrended = initial - (level + trend * (np.arange(warmup) - warmup + 1))
    seasonal = detrended.reshape(-1, p).mean(axis=0)
    for t in range(end, len(values)):
        phase = (t - start) % p
        if t > end:
            value = values[t]
            if np.isnan(value):
                level += trend
            else:
                previous_level = level
                level = params['alpha'] * (value - seasonal[phase]) + (1 - params['alpha']) * (level + trend)
                if with_trend:
                    trend = params['beta'] * (level - previous_level) + (1 - params['beta']) * trend
                seasonal[phase] = params['gamma'] * (value - level) + (1 - params['gamma']) * seasonal[phase]
        forecast[t + h] = level + h * trend + seasonal[(phase + h) % p]
    return forecast


def AdditiveWintersExponentialSmoothing(x, h, Params):
    """Additive level/seasonality model; first p observations initialize it.

    Forecasts before initialization are NaN. After initialization a missing
    observation leaves the state unchanged; the seasonal calendar still advances.
    """
    return _seasonal_smoothing(x, h, Params, with_trend=False)


def TheilWageExponentialSmoothing(x, h, Params):
    """Additive level/trend/seasonality model with forecast l + h*b + s.

    Two complete seasons initialize level, slope and centered seasonal effects.
    Earlier forecasts are NaN. Missing observations propagate the trend while
    retaining the seasonal effects. Input observations must be equally spaced.
    """
    return _seasonal_smoothing(x, h, Params, with_trend=True)


# AdaptiveExponentialSmoothing
# x <array Tx1>- time series, 
# h <scalar> - forecasting delay
# Params <dict> - dictionary with 
#    alpha <scalar in [0,1]> - smoothing parameter
#    AdaptivePeriod scalar> - adapation period for initialization
#    gamma<scalar in [0,1]> - parametr of cross validation
def AdaptiveExponentialSmoothing(x, h, Params):
    x = _smoothing_input(x, h, Params, ['alpha', 'gamma'])
    T = len(x)
    alpha = Params['alpha']
    gamma = Params['gamma']
    AdaptationPeriod = _positive_int(Params['AdaptationPeriod'], 'AdaptationPeriod')
    FORECAST = [np.nan] * (T + h)
    y = np.nan
    t0= np.nan
    e1= np.nan
    e2= np.nan
    Kt_1 = alpha
    K=alpha
    for t in range(0, T):
        if not math.isnan(x[t]):
            if math.isnan(y):
                y=x[t]
                t0=t
                e1=alpha
                e2 = 1
            else:
                if (t-t0)<h:
                    e1 = gamma*(x[t]-y)+(1-gamma)*e1
                    e2 = gamma*np.abs(x[t]-y)+(1-gamma)*e2
                else:
                    e1 = gamma*(x[t]-FORECAST[t])+(1-gamma)*e1
                    e2 = gamma*np.abs(x[t]-FORECAST[t])+(1-gamma)*e2
            
            if e2==0:
                K=alpha
            else:
                K=np.abs(e1/e2)

            alpha=Kt_1
            Kt_1=K

            if (t-t0+1)<AdaptationPeriod:
                y = y*(1-alpha)*(t-t0+1)/(AdaptationPeriod) + (1-(1-alpha)*(t-t0+1)/(AdaptationPeriod))*x[t]
            else:
                y = y*(1-alpha) + (alpha)*x[t]
        FORECAST[t+h] = y
    return FORECAST

# generate forecast values based on particular algorithm
# h - forecast horizon, each point in historical period will be forecasted with delay = h (h-step ahead)
# ts - <pandas data frame> with timestamps in index, each column contains particular timeseries, all of them will be forecasted independently
# AlgName - <str> name of the function that runs forecasting algorithm 
# AlgTitle <str> - a name of the forecasting algorithm
# step <char> - aggregation method of the original data before forecasting
# ParamsArray <array> - array of parameter set, each component of array defines particular forecasting algorithm

def build_forecast(h, ts, alg_name, alg_title, params, step='D'):
  'grid'

  FRC_TS = dict()

  for p in params:
      frc_horizon = pd.date_range(ts.index[-1], periods=h+1, freq=step)[1:]
      frc_ts = pd.DataFrame(index = ts.index.append(frc_horizon), columns = ts.columns)

      for cntr in ts.columns:
          frc_ts[cntr] = eval(alg_name)(ts[cntr], h, p)

#         frc_ts.columns = frc_ts.columns+('%s %s' % (alg_title, p))
      FRC_TS['%s %s' % (alg_title, p)] = frc_ts

  return FRC_TS

# draw forecast and original time series
# ts - <pandas data frame> with timestamps in index, each column contains particular timeseries
# frc_ts - <pandas data frame> the same structure as ts, 
# ts_num <int> - column index for which plot shoud be drawn
# alg_title <str> - a name of the forecasting algorithm
def plot_ts_forecast(ts, frc_ts, ts_num=0, alg_title='', title_text = ''):
    frc_ts.columns = ts.columns+'; '+alg_title
    ts[[ts.columns[ts_num]]].merge(frc_ts[[frc_ts.columns[ts_num]]], how = 'outer', left_index = True, right_index = True)\
      .plot().update_layout(height=350, width=1300,
                  xaxis_title="time ticks",
                  yaxis_title="ts and forecast values", title_text=title_text, ).show()
    return
# deprecated: matplotlib version
#def plot_ts_forecast(ts, frc_ts, ts_num=0, alg_title=''):
#	frc_ts.columns = ts.columns+'; '+alg_title
#	ts[ts.columns[ts_num]].plot(style='b', linewidth=1.0, marker='o')
#	ax = frc_ts[frc_ts.columns[ts_num]].plot(style='r-^', figsize=(25,5), linewidth=1.0)
#	plt.xlabel("Time ticks")
#	plt.ylabel("TS values")
#	plt.legend()
#	return ax


def draw_arima_forecast(ts, arima_model, start_dt=0, end_dt=-1):
  predict = arima_model.get_prediction()
  forecast = pd.DataFrame(predict.predicted_mean).rename(columns = {'predicted_mean':'static_forecast'})
  forecast_ci = predict.conf_int().rename(columns = {'lower wage_boxcox':'l_ci_st',	'upper wage_boxcox':'u_ci_st'}) # confidence interval

  #  Dynamic predictions
  predict_dy = arima_model.get_prediction(dynamic=start_dt)
  forecast_dy = pd.DataFrame(predict_dy.predicted_mean).rename(columns = {'predicted_mean':'dynamic_forecast'})
  forecast_dy_ci = predict_dy.conf_int().rename(columns = {'lower wage_boxcox':'l_ci_dy',	'upper wage_boxcox':'u_ci_dy'}) # confidence interval

  if start_dt ==0:
    start_dt= ts.index.min()

  if end_dt == -1:
    end_dt = ts_wage_boxcox.index.max()

  # Plot data points and predictions
  fig = ts.loc[start_dt:].merge(
      forecast[start_dt:end_dt],
        how = 'left', left_index = True, right_index = True
        ).merge(
            forecast_dy[start_dt:],
            how = 'left', left_index = True, right_index = True
        ).merge(
          forecast_ci,
          how = 'left', left_index = True, right_index = True
        ).merge(
          forecast_dy_ci,
          how = 'left', left_index = True, right_index = True
        ).plot().update_layout(height=350, width=1300).show()

  return fig

# Quality functions: return (aggregate, per-observation errors/contributions).
# Ratios are fractions, not percentages. DataFrames are scored per column.
def _metric_inputs(x, y):
    if not isinstance(x, (pd.Series, pd.DataFrame)):
        x = pd.Series(x, dtype=float)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        if type(x) is not type(y) or not x.index.equals(y.index):
            raise ValueError('Actuals and forecasts must have matching types and indexes')
        if isinstance(x, pd.DataFrame) and not x.columns.equals(y.columns):
            raise ValueError('Actuals and forecasts must have matching columns')
    elif isinstance(x, pd.DataFrame):
        y = pd.DataFrame(y, index=x.index, columns=x.columns, dtype=float)
    else:
        y = pd.Series(y, index=x.index, dtype=float)
    x, y = x.astype(float), y.astype(float)
    valid = np.isfinite(x) & np.isfinite(y)
    return x.where(valid), y.where(valid)


def _ratio(errors, denominator):
    return (errors / denominator).replace([np.inf, -np.inf], np.nan)


def qualitySSE(x, y):
    x, y = _metric_inputs(x, y)
    errors = (x - y) ** 2
    return errors.sum(min_count=1), errors


def qualityMSE(x, y):
    x, y = _metric_inputs(x, y)
    errors = (x - y) ** 2
    return errors.mean(), errors


def qualityRMSE(x, y):
    x, y = _metric_inputs(x, y)
    errors = (x - y).abs()
    return ((errors ** 2).mean()) ** 0.5, errors


def qualityMAE(x, y):
    x, y = _metric_inputs(x, y)
    errors = (x - y).abs()
    return errors.mean(), errors


def qualityMAPE(x, y):
    """Zero actuals are undefined (NaN) and excluded from the mean."""
    x, y = _metric_inputs(x, y)
    errors = _ratio((x - y).abs(), x.abs())
    return errors.mean(), errors


def qualityMAPPE(x, y):
    """Zero forecasts are undefined (NaN) and excluded from the mean."""
    x, y = _metric_inputs(x, y)
    errors = _ratio((x - y).abs(), y.abs())
    return errors.mean(), errors


def qualitySMAPE(x, y):
    """Symmetric absolute percentage error in [0, 2]; define 0/0 as 0."""
    x, y = _metric_inputs(x, y)
    denominator = x.abs() + y.abs()
    errors = _ratio(2 * (x - y).abs(), denominator).mask(denominator == 0, 0.0)
    return errors.mean(), errors


def qualityMAMAXPE(x, y):
    """Use the elementwise maximum absolute value; define 0/0 as 0."""
    x, y = _metric_inputs(x, y)
    denominator = np.maximum(x.abs(), y.abs())
    errors = _ratio((x - y).abs(), denominator).mask(denominator == 0, 0.0)
    return errors.mean(), errors


def qualityMASE(x, y, init_step=0, *, train, seasonality=1):
    """Mean absolute error scaled by TRAINING seasonal-naive MAE.

    Pass training observations explicitly: qualityMASE(actual, forecast,
    train=history, seasonality=7). The default seasonality=1 is non-seasonal.
    init_step excludes leading evaluation rows, never training observations.
    Constant/seasonally constant training data give an undefined scale (NaN).
    Missing training pairs are omitted without closing calendar gaps.
    """
    m = _positive_int(seasonality, 'seasonality')
    if isinstance(init_step, (bool, np.bool_)) or not isinstance(init_step, (int, np.integer)) or init_step < 0:
        raise ValueError('init_step must be a non-negative integer')
    x, y = _metric_inputs(x, y)
    if isinstance(x, pd.DataFrame):
        if not isinstance(train, pd.DataFrame) or not x.columns.equals(train.columns):
            raise ValueError('Training data must have the same columns as actuals')
        history = train.astype(float)
    else:
        history = pd.Series(train, dtype=float)
    if len(history) <= m:
        raise ValueError('Training data must contain more than seasonality observations')
    history = history.where(np.isfinite(history))
    scale = (history - history.shift(m)).abs().mean()
    errors = _ratio((x - y).abs().iloc[init_step:], scale)
    return errors.mean(), errors


def qualityMedianAE(x, y):
    x, y = _metric_inputs(x, y)
    errors = (x - y).abs()
    return errors.median(), errors


def _weighted_error(x, y, denominator):
    contributions = _ratio((x - y).abs(), denominator.sum(min_count=1))
    return contributions.sum(min_count=1), contributions


def qualityWAPE(x, y):
    """Undefined for zero total actual volume; return NaN, including all-zero pairs."""
    x, y = _metric_inputs(x, y)
    return _weighted_error(x, y, x.abs())


def qualityWAPPE(x, y):
    """Undefined for zero total forecast volume; return NaN."""
    x, y = _metric_inputs(x, y)
    return _weighted_error(x, y, y.abs())


def qualityWAMAXPE(x, y):
    """Scale by the sum of elementwise absolute maxima; zero total gives NaN."""
    x, y = _metric_inputs(x, y)
    return _weighted_error(x, y, np.maximum(x.abs(), y.abs()))


def get_autoregrmatrix(x,h,K):
    T = len(x)
    X = sc.linalg.hankel(x[:T-h-K+1], 
                          np.hstack((x[T-h-K:T-h]))) # is needed to repeat x[-K] in second part
    y = x[K+h-1:]
    return X,y
