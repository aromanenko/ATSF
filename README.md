# ATSF
Applied Time Series Forecasting course materials

## Forecast and metric conventions

The reference implementations live in `utils.py`. Lesson 7 imports the metrics
from this module; lesson 3 displays and uses the shared smoothing recurrences.
When opening a notebook in Colab, upload `utils.py` alongside the notebook.

- Smoothing output at position `t + h` is the forecast made using observations
  through `t`. This is a rolling h-step-ahead forecast, not a full forecast path
  from one fixed origin. Input observations must be equally spaced.
- Holt initializes its level from the first observation and its trend from the
  second available observation. Its adaptation weights approach `alpha` and
  `beta` over `AdaptationPeriod` ticks. Invalid coefficients or horizons raise
  `ValueError`.
- Additive Winters needs one complete season; Theil–Wage needs two. Predictions
  before initialization are `NaN`. After initialization, missing observations
  retain seasonal effects and propagate the trend. Compare models on a common
  evaluation period after all models have initialized.
- Metrics return `(aggregate, per_observation_values)` and score DataFrames per
  column. Ratios are fractions, not percentages. Pandas indexes and columns must
  match; array forecasts are matched positionally. Non-finite pairs are omitted.
- MAPE at zero actuals and MAPPE at zero forecasts are undefined and omitted.
  Report the number of valid evaluation pairs when using these metrics.
- SMAPE uses absolute values in its denominator. SMAPE and MAMAXPE define a
  zero/zero observation as zero error. Weighted metrics with zero total volume,
  and MASE with zero training scale, return `NaN`, including perfect all-zero
  forecasts. An all-missing evaluation also returns `NaN`.

MASE requires a separate training history. Its scale is the mean absolute
seasonal-naive error **on that history**, with `seasonality=1` for non-seasonal
series. `init_step` only skips leading evaluation rows:

```python
import pandas as pd
from utils import qualityMASE

score, errors = qualityMASE(
    pd.Series([20., 30.]),
    pd.Series([18., 26.]),
    train=pd.Series([2., 4., 6., 10.]),
    seasonality=1,
)
assert score == 1.125
```

## Reference implementation checks

With Python, NumPy, pandas, SciPy, Plotly and pytest installed, run from the
repository root:

```sh
python -m pytest tests -q
```

The checks cover hand-calculated metrics, zero/missing values, constant and
linear series, seasonal phase alignment at multiple horizons, initial adaptation,
and independence from future observations. They also check the corrected
notebook examples against the shared implementations. Saved outputs in lessons
2, 3 and 7 have been cleared because they predate these corrections; rerun those
notebooks to regenerate their plots and scores.
