import pandas as pd, numpy as np,datetime

IB_ZERO_DEMAND_THRESHOLD = 0.001


def generate_data(PERIOD_DT: datetime.datetime, PERIOD_END_DT: datetime.datetime) -> pd.DataFrame:
    """
    Function generating input data

    Parameters
    ----------
    PERIOD_DT : datetime.datetime
        First known date
    PERIOD_END_DT : datetime.datetime
        Last known date (i.e. sales and stock information is known)

    Returns
    -------
    pd.DataFrame
        input data of the algorithm
	"""

    date = pd.date_range(PERIOD_DT, PERIOD_END_DT, freq='D')
    size = len(date)

    VF_FORECAST_VALUE = abs(np.random.normal(size=size))
    ML_FORECAST_VALUE = abs(np.random.normal(size=size))
    MID_RECONCILED_FORECAST = abs(np.random.normal(size=size))
    DEMAND_TYPE = np.random.choice(['promo', 'regular'], size=size)
    ASSORTMENT_TYPE = np.random.choice(['new', 'old'], size=size)
    PRODUCT_LVL_ID1 = np.random.choice([10084, 10091, 12121, 12311], size=size)
    LOCATION_LVL_ID = np.full(size, 10000)
    HYBRID_FORECAST_VALUE = abs(np.random.normal(size=size))
    DISTR_CHANNEL_LVL_ID = np.random.choice([1, 2, 3, 4], size=size)
    CUSTOMER_LVL_ID1 = np.random.choice(range(900000, 1000000), size=size)
    SEGMENT_NAME = np.random.choice(['Low Volume', 'Retired', 'Short'], size=size)

    df = pd.concat([pd.Series(i) for i in
                    [date, SEGMENT_NAME, HYBRID_FORECAST_VALUE, CUSTOMER_LVL_ID1, DISTR_CHANNEL_LVL_ID,
                     VF_FORECAST_VALUE, ML_FORECAST_VALUE,
                     MID_RECONCILED_FORECAST, DEMAND_TYPE, ASSORTMENT_TYPE,
                     PRODUCT_LVL_ID1, LOCATION_LVL_ID, ]], axis=1)
    df.columns = ['date', 'SEGMENT_NAME', 'HYBRID_FORECAST_VALUE', 'CUSTOMER_LVL_ID1', 'DISTR_CHANNEL_LVL_ID',
                  'VF_FORECAST_VALUE', 'ML_FORECAST_VALUE',
                  'MID_RECONCILED_FORECAST', 'DEMAND_TYPE', 'ASSORTMENT_TYPE',
                  'PRODUCT_LVL_ID1', 'LOCATION_LVL_ID']
    return df


def hybridization(data: pd.DataFrame) -> pd.DataFrame:
    """
    Function generating input data

    Parameters
    ----------
    data : pd.Dataframe
        Input table 

    Returns
    -------
    pd.DataFrame
        Transformed table with hybridized forecasts
    """


    if 'ML_FORECAST_VALUE' not in data:
        data['ML_FORECAST_VALUE_F'] = data.VF_FORECAST_VALUE

    if 'VF_FORECAST_VALUE' not in data.columns:
        data['VF_FORECAST_VALUE_F'] = data.ML_FORECAST_VALUE

    HYBRID_FORECAST_VALUE = []
    FORECAST_SOURCE = []
    ENSEMBLE_FORECAST_VALUE = []


    for _, row in data.iterrows():
        if (row.DEMAND_TYPE.lower() == 'promo' and row.SEGMENT_NAME.lower() != 'retired') or row.SEGMENT_NAME.lower() == 'short' or \
                row.ASSORTMENT_TYPE.lower() == 'new':
            HYBRID_FORECAST_VALUE.append(row.ML_FORECAST_VALUE)
            FORECAST_SOURCE.append('ml')
            ENSEMBLE_FORECAST_VALUE.append(np.nan)
        elif row.SEGMENT_NAME.lower() == 'retired' or row.SEGMENT_NAME.lower() == 'low volume':
            VF_FORECAST_VALUE_F = IB_ZERO_DEMAND_THRESHOLD
            HYBRID_FORECAST_VALUE.append(VF_FORECAST_VALUE_F)
            FORECAST_SOURCE.append('vf')
            ENSEMBLE_FORECAST_VALUE.append(np.nan)
        else:
            HYBRID_FORECAST_VALUE.append(np.mean(row.ML_FORECAST_VALUE, row.VF_FORECAST_VALUE))
            FORECAST_SOURCE.append('ensemble')
            ENSEMBLE_FORECAST_VALUE.append(np.mean(row.ML_FORECAST_VALUE, row.VF_FORECAST_VALUE))
    HYBRID_FORECAST_VALUE = pd.Series(HYBRID_FORECAST_VALUE)
    FORECAST_SOURCE = pd.Series(FORECAST_SOURCE)
    ENSEMBLE_FORECAST_VALUE = pd.Series(ENSEMBLE_FORECAST_VALUE)
    data['HYBRID_FORECAST_VALUE'] = HYBRID_FORECAST_VALUE
    data['FORECAST_SOURCE'] = FORECAST_SOURCE
    data['ENSEMBLE_FORECAST_VALUE'] = ENSEMBLE_FORECAST_VALUE

    return data