import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import utils as u


def test_mase_uses_training_scale_not_evaluation_differences():
    # Training naive errors = 2, 2, 4; test absolute errors = 2, 4.
    score, errors = u.qualityMASE([20, 30], [18, 26], train=[2, 4, 6, 10])
    assert score == pytest.approx(1.125)
    np.testing.assert_allclose(errors, [0.75, 1.5])
    shifted, _ = u.qualityMASE([120, 130], [118, 126], train=[2, 4, 6, 10])
    assert shifted == score


def test_seasonal_mase_and_initial_evaluation_exclusion():
    # Seasonal naive errors at m=2 = 2, 4, 2, 4; scale = 3.
    score, errors = u.qualityMASE([99, 20, 30], [0, 17, 24], init_step=1,
                                 train=[10, 20, 12, 24, 14, 28], seasonality=2)
    assert score == 1.5
    np.testing.assert_allclose(errors, [1, 2])
    assert errors.index.tolist() == [1, 2]


@pytest.mark.parametrize('train,m', [([5, 5, 5], 1), ([1, 3, 1, 3], 2), ([np.nan]*3, 1)])
def test_undefined_mase_scale(train, m):
    score, errors = u.qualityMASE([5, 6], [5, 4], train=train, seasonality=m)
    assert np.isnan(score)
    assert errors.isna().all()


def test_mase_requires_explicit_history():
    with pytest.raises(TypeError):
        u.qualityMASE([1, 2], [2, 3])
    with pytest.raises(ValueError):
        u.qualityMASE([1, 2], [2, 3], train=[1, 2], seasonality=2)


def test_mase_dataframe_per_series_scales():
    train = pd.DataFrame({'a': [0, 2, 4], 'b': [0, 4, 8]})
    actual = pd.DataFrame({'a': [10, 20], 'b': [10, 20]})
    score, _ = u.qualityMASE(actual, actual - 2, train=train)
    np.testing.assert_allclose(score, [1, .5])


def test_mase_missing_history_does_not_close_calendar_gaps():
    score, _ = u.qualityMASE([10], [8], train=[1, np.nan, 5, 7])
    assert score == 1


@pytest.mark.parametrize('metric', [u.qualitySMAPE, u.qualityMAMAXPE])
def test_symmetric_metrics_signed_values_and_zeros(metric):
    score, errors = metric([-2, 0, 2], [2, 0, -2])
    np.testing.assert_allclose(errors, [2, 0, 2])
    assert score == pytest.approx(4/3)


def test_max_percentage_metrics_are_elementwise():
    actual = pd.Series([-2, 4, 0])
    predicted = pd.Series([1, -2, 3])
    score, errors = u.qualityMAMAXPE(actual, predicted)
    np.testing.assert_allclose(errors, [1.5, 1.5, 1])
    assert score == pytest.approx(4/3)
    assert u.qualityWAMAXPE(actual, predicted)[0] == pytest.approx(12/9)


@pytest.mark.parametrize('metric', [u.qualityMAPE, u.qualityMAPPE, u.qualityWAPE, u.qualityWAPPE, u.qualityWAMAXPE])
def test_undefined_zero_denominators_are_not_reported_as_perfect(metric):
    score, errors = metric([0, 0], [0, 0])
    assert np.isnan(score)
    assert errors.isna().all()


def test_percentage_missing_pair_policy():
    assert u.qualityWAPE([10, 100], [8, np.nan])[0] == .2
    score, errors = u.qualityMAPE([0, 10], [5, 8])
    assert score == .2
    assert np.isnan(errors.iloc[0])


@pytest.mark.parametrize('metric', [u.qualitySSE, u.qualityMSE, u.qualityRMSE, u.qualityMAE,
                                   u.qualityMedianAE, u.qualitySMAPE, u.qualityWAPE])
def test_all_missing_is_undefined(metric):
    assert np.isnan(metric([np.nan, np.nan], [1, 2])[0])


def test_metrics_do_not_silently_align_mismatched_dates():
    with pytest.raises(ValueError, match='indexes'):
        u.qualityMAE(pd.Series([1, 2], index=[0, 1]), pd.Series([1, 2], index=[1, 2]))


@pytest.mark.parametrize('metric,expected', [(u.qualitySSE, 5), (u.qualityMSE, 2.5),
    (u.qualityRMSE, np.sqrt(2.5)), (u.qualityMAE, 1.5), (u.qualityMedianAE, 1.5)])
def test_basic_metrics_hand_calculation(metric, expected):
    assert metric([1, 3], [2, 1])[0] == pytest.approx(expected)


MODELS = [
    (u.InitExponentialSmoothing, {'alpha': .3, 'AdaptationPeriod': 4}, 0),
    (u.HoltExponentialSmoothing, {'alpha': .3, 'beta': .2, 'AdaptationPeriod': 4}, 0),
    (u.AdditiveWintersExponentialSmoothing, {'alpha': .3, 'gamma': .2, 'seasonality_period': 4}, 3),
    (u.TheilWageExponentialSmoothing, {'alpha': .3, 'beta': .2, 'gamma': .2, 'seasonality_period': 4}, 7),
    (u.AdaptiveExponentialSmoothing, {'alpha': .3, 'gamma': .2, 'AdaptationPeriod': 4}, 0),
]


@pytest.mark.parametrize('model,params,warmup', MODELS)
@pytest.mark.parametrize('h', [1, 4, 8])
def test_constant_series_and_output_alignment(model, params, warmup, h):
    result = np.asarray(model([10.] * 24, h, params))
    assert len(result) == 24 + h
    assert np.isnan(result[:warmup + h]).all()
    np.testing.assert_allclose(result[warmup + h:], 10.)


@pytest.mark.parametrize('h', [1, 4, 9])
def test_holt_linear_trend_exact_after_two_observations(h):
    result = u.HoltExponentialSmoothing(3 + 2 * np.arange(20), h,
                                      {'alpha': .3, 'beta': .2, 'AdaptationPeriod': 5})
    np.testing.assert_allclose(result[h+1:], 3 + 2 * np.arange(h+1, 20+h))


@pytest.mark.parametrize('h', [1, 4, 8, 9])
def test_seasonal_phase_at_arbitrary_horizon(h):
    seasonal = np.array([-3, 1, -1, 3])
    times = np.arange(24)
    result = u.AdditiveWintersExponentialSmoothing(10 + seasonal[times % 4], h,
                  {'alpha': .3, 'gamma': .2, 'seasonality_period': 4})
    np.testing.assert_allclose(result[3+h:], 10 + seasonal[np.arange(3+h, 24+h) % 4])


@pytest.mark.parametrize('h', [1, 4, 8, 9])
def test_theil_wage_linear_trend_plus_seasonality(h):
    seasonal = np.array([-3, 1, -1, 3])
    times = np.arange(24)
    values = 10 + 2*times + seasonal[times % 4]
    result = u.TheilWageExponentialSmoothing(values, h,
                  {'alpha': .3, 'beta': .2, 'gamma': .2, 'seasonality_period': 4})
    future = np.arange(7+h, 24+h)
    np.testing.assert_allclose(result[7+h:], 10 + 2*future + seasonal[future % 4], atol=1e-12)


def test_holt_adaptation_weights_hand_calculation():
    params = {'alpha': .2, 'beta': .4, 'AdaptationPeriod': 6}
    # At t=2: alpha_t=.6, beta_t=.7; previous level=2, trend=1.
    # level=.6*8+.4*3=6; trend=.7*(6-2)+.3*1=3.1.
    assert u.HoltExponentialSmoothing([1, 2, 8], 2, params)[4] == pytest.approx(12.2)
    without = u.HoltExponentialSmoothing([1, 2, 8], 2, dict(params, AdaptationPeriod=1))
    assert without[4] != pytest.approx(12.2)


def test_initialized_ses_performs_one_update_per_observation():
    params = {'alpha': .2, 'AdaptationPeriod': 4}
    # At t=1: alpha_t=1-.8*(2/4)=.6; level=.6*10+.4*0=6.
    assert u.InitExponentialSmoothing([0, 10], 1, params)[2] == pytest.approx(6)


@pytest.mark.parametrize('model,params,warmup', MODELS)
def test_forecasts_never_change_when_future_observations_change(model, params, warmup):
    values = np.array([1, 3, 2, 4, 2, 4, 3, 5, 4, 7, 4, 8.])
    h = 4
    complete = np.asarray(model(values, h, params))
    for cutoff in range(1, len(values)):
        prefix = model(values[:cutoff], h, params)
        np.testing.assert_allclose(prefix, complete[:cutoff+h], equal_nan=True)


@pytest.mark.parametrize('model,params,warmup', MODELS)
def test_validation_missing_and_positional_indexing(model, params, warmup):
    values = pd.Series([np.nan] + [10.] * 16 + [np.nan, 10.], index=np.arange(100, 119))
    result = np.asarray(model(values, 4, params))
    np.testing.assert_allclose(result[np.isfinite(result)], 10.)
    assert np.isnan(model([np.nan] * 8, 1, params)).all()
    for bad in [-.1, 1.1, np.nan]:
        with pytest.raises(ValueError, match='alpha'):
            model([1, 2, 3], 1, dict(params, alpha=bad))
    with pytest.raises(ValueError, match='h'):
        model([1, 2, 3], 0, params)
    with pytest.raises(ValueError, match='non-empty'):
        model([], 1, params)


def test_ses_datetime_index_and_missing_observations():
    x = pd.Series([np.nan, 10, 10, np.nan], index=pd.date_range('2020-01-01', periods=4))
    result = u.SimpleExponentialSmoothing(x, h=3)
    assert result.index.equals(pd.date_range('2020-01-01', periods=7))
    np.testing.assert_allclose(result.iloc[4:], 10.)
    with pytest.raises(ValueError, match='alpha'):
        u.SimpleExponentialSmoothing(x, params={'alpha': 2})


def test_holt_gap_propagates_trend():
    result = u.HoltExponentialSmoothing([1, 3, np.nan, 7], 2,
                                      {'alpha': .3, 'beta': .2, 'AdaptationPeriod': 4})
    np.testing.assert_allclose(result[3:], [7, 9, 11])


def test_lesson7_uses_shared_metrics_and_hand_checks_run():
    notebook = json.loads((Path(__file__).parents[1] / '7_Loss Functions in Forecasting.ipynb').read_text())
    namespace = {'np': np, 'pd': pd}
    for cell in notebook['cells']:
        text = ''.join(cell['source'])
        if text.startswith('# One tested') or cell['metadata'].get('id') == 'metric-hand-checks':
            exec(text, namespace)
    assert namespace['qualityMASE'] is u.qualityMASE
    assert namespace['mase'] == pytest.approx(1.125)


def test_lesson3_wrappers_use_corrected_models():
    import ast
    notebook = json.loads((Path(__file__).parents[1] / '3_Holt, Winters, Theil-Wage.ipynb').read_text())
    namespace = dict(vars(u))
    for cell in notebook['cells']:
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and 'def my' in text:
            definitions = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), '<lesson3>', 'exec'), namespace)
    x = pd.Series(10 + np.arange(24), index=pd.date_range('2020-01-01', periods=24))
    for model, params, _ in MODELS[1:4]:
        result = namespace['my' + model.__name__](x, 8, params)
        np.testing.assert_allclose(result, model(x, 8, params), equal_nan=True)
        assert result.index.equals(pd.date_range('2020-01-01', periods=32))


def test_lesson2_reference_ses_matches_shared_implementation():
    import ast
    import math
    notebook = json.loads((Path(__file__).parents[1] / '2_Simple Exponential Smothing.ipynb').read_text())
    namespace = {'np': np, 'pd': pd, 'math': math}
    for cell in notebook['cells']:
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and any('def '+name+'(' in text for name in [
                'SimpleExponentialSmoothing', 'InitExponentialSmoothing', 'AdaptiveExponentialSmoothing']):
            definitions = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), '<lesson2>', 'exec'), namespace)
    x = pd.Series([1., 3., np.nan, 8., 4.], index=pd.date_range('2020-01-01', periods=5), name='x')
    for name, params in [('SimpleExponentialSmoothing', {'alpha': .2}),
                         ('InitExponentialSmoothing', {'alpha': .2, 'AdaptationPeriod': 4}),
                         ('AdaptiveExponentialSmoothing', {'alpha': .2, 'gamma': .1, 'AdaptationPeriod': 4})]:
        result = namespace[name](x, 3, params)
        np.testing.assert_allclose(result, getattr(u, name)(x, 3, params), equal_nan=True)
        with pytest.raises(ValueError, match='alpha'):
            namespace[name](x, 3, dict(params, alpha=2))
