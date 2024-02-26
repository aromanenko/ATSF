import pandas as pd
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')

def generate_data(DR_PARAMETERS : dict, TGT_VAR_CONFIG : dict,
                  IB_MAX_DT : datetime.datetime, IB_UPDATE_HISTORY_DEPTH : int,
                 IB_HIST_START_DT : datetime.datetime, IB_HIST_END_DT : datetime.datetime,
                 hierarchies : dict, STOCK : pd.DataFrame, PROMO : pd.DataFrame,
                  PRODUCT_ATTR : pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Function generating input data
    
    Parameters
    ----------
    DR_PARAMETERS : dict
        Configuration parameters used within the step
    TGT_VAR_CONFIG : dict
        Configuration parameters used within the step
    IB_MAX_DT
        Maximal date which is used for data preparation (e.g. 01/01/2100)
    IB_UPDATE_HISTORY_DEPTH : int
        Number of days of historical information that should be considered within this step running,
        i.e. only dates since (>=) IB_HIST_END_DT -  IB_UPDATE_HISTORY_DEPTH should be used within step
    IB_HIST_START_DT : datetime.datetime
        Minimal date that should be present in Demand Restored
    IB_HIST_END_DT : datetime.datetime
        Last known date (i.e. sales and stock information is known)
    hierarchies : dict
        Dictionary containg matches of key names with the relevant hierarchical tables
    STOCK : pd.DataFrame
        STOCK table
    PROMO : pd.DataFrame
        PROMO table
    PRODUCT_ATTR : pd.DataFrame
        PRODUCT_ATTR table
       
    Returns
    -------
    pd.DataFrame
        SALES table
    pd.DataFrame
        FORECAST_FLAG table
    pd.DataFrame
        RESTORED_DEMAND table
    pd.DataFrame
        PROMO table
    pd.DataFrame
        STOCK table
    """
    PRODUCT = hierarchies['PRODUCT']
    LOCATION = hierarchies['LOCATION']
    CUSTOMER = hierarchies['CUSTOMER']
    DISTR_CHANNEL = hierarchies['DISTR_CHANNEL']
    SALES = pd.DataFrame()
    SALES['PERIOD_DT'] = pd.date_range(IB_HIST_START_DT, IB_HIST_END_DT)
    product_df = pd.DataFrame(PRODUCT['PRODUCT_ID']).drop_duplicates()
    SALES = pd.merge(SALES, product_df, 'cross')
    location_df = pd.DataFrame(LOCATION['LOCATION_ID']).drop_duplicates()
    SALES = pd.merge(SALES, location_df, 'cross')
    customer_df = pd.DataFrame(CUSTOMER['CUSTOMER_ID']).drop_duplicates()
    SALES = pd.merge(SALES, customer_df, 'cross')
    distr_channel_df = pd.DataFrame(DISTR_CHANNEL['DISTR_CHANNEL_ID']).drop_duplicates()
    SALES = pd.merge(SALES, distr_channel_df, 'cross')
    SALES['SALES_QTY'] = np.random.randint(1000, size=SALES.shape[0])
    SALES['PROMO_FLG'] = np.random.choice([0, 1], p=[0.8, 0.2], size=SALES.shape[0])
    SALES.loc[SALES['PROMO_FLG'] == 1, 'PROMO_ID'] = np.random.choice(PROMO['PROMO_ID'].tolist())

    FORECAST_FLAG = pd.DataFrame()
    FORECAST_FLAG['PERIOD_START_DT'] = [IB_HIST_START_DT]
    FORECAST_FLAG['PERIOD_END_DT'] = [IB_HIST_START_DT]

    for key in hierarchies:
        key_df = hierarchies[key][f"{key}_ID"]
        FORECAST_FLAG = pd.merge(FORECAST_FLAG, key_df, 'cross')

    FORECAST_FLAG['STATUS'] = np.random.choice(['maturity', 'new', 'end-of-life'], size=FORECAST_FLAG.shape[0])
    FORECAST_FLAG.loc[(FORECAST_FLAG['PRODUCT_ID'] == PROMO['PRODUCT_ID'][0]) &
                     (FORECAST_FLAG['LOCATION_ID'] == PROMO['LOCATION_ID'][0]) &
                     (FORECAST_FLAG['CUSTOMER_ID'] == PROMO['CUSTOMER_ID'][0]) &
                     (FORECAST_FLAG['DISTR_CHANNEL_ID'] == PROMO['DISTR_CHANNEL_ID'][0]), 'STATUS'] = 'maturity'

    RESTORED_DEMAND = pd.DataFrame()
    RESTORED_DEMAND['PERIOD_DT'] = pd.date_range(IB_HIST_START_DT, IB_HIST_END_DT)
    RESTORED_DEMAND = pd.merge(RESTORED_DEMAND, PRODUCT['PRODUCT_ID'], 'cross')
    RESTORED_DEMAND = pd.merge(RESTORED_DEMAND, LOCATION['LOCATION_ID'], 'cross')
    RESTORED_DEMAND = pd.merge(RESTORED_DEMAND, CUSTOMER['CUSTOMER_ID'], 'cross')
    RESTORED_DEMAND = pd.merge(RESTORED_DEMAND, DISTR_CHANNEL['DISTR_CHANNEL_ID'], 'cross')
    RESTORED_DEMAND['TGT_QTY'] = abs(np.random.normal(500, 300, RESTORED_DEMAND.shape[0]))
    RESTORED_DEMAND['TGT_QTY_R'] = RESTORED_DEMAND['TGT_QTY'] + np.random.normal(50, 30, RESTORED_DEMAND.shape[0])
    RESTORED_DEMAND['STOCK_QTY'] = abs(np.random.normal(500, 300, RESTORED_DEMAND.shape[0]))
    RESTORED_DEMAND['PROMO_FLG'] = np.random.choice([0, 1], p=[0.8, 0.2], size=RESTORED_DEMAND.shape[0])
    RESTORED_DEMAND['PROMO_TYPE'] = np.nan
    RESTORED_DEMAND.loc[(RESTORED_DEMAND['PRODUCT_ID'] == PROMO['PRODUCT_ID'][0]) &
                     (RESTORED_DEMAND['LOCATION_ID'] == PROMO['LOCATION_ID'][0]) &
                     (RESTORED_DEMAND['CUSTOMER_ID'] == PROMO['CUSTOMER_ID'][0]) &
                     (RESTORED_DEMAND['DISTR_CHANNEL_ID'] == PROMO['DISTR_CHANNEL_ID'][0]), 'PROMO_FLG'] = 1

    RESTORED_DEMAND.loc[RESTORED_DEMAND['PROMO_FLG'] == 1, 'PROMO_TYPE'] = np.random.choice(
        PROMO['PROMO_TYPE'].tolist(), size=(RESTORED_DEMAND['PROMO_FLG'] == 1).sum()
    )

    RESTORED_DEMAND = pd.merge(RESTORED_DEMAND, PROMO[['PROMO_ID', 'PROMO_TYPE']], on='PROMO_TYPE', how='left')

    RESTORED_DEMAND['DEFICIT_FLG1'] = np.random.choice([0, 1], p=[0.8, 0.2], size=RESTORED_DEMAND.shape[0])
    RESTORED_DEMAND['DEFICIT_FLG2'] = np.random.choice([0, 1], p=[0.8, 0.2], size=RESTORED_DEMAND.shape[0])

    #group_cols = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']

    #RESTORED_DEMAND['DEFICIT_FLG1'] = 1 - RESTORED_DEMAND['DEFICIT_FLG1']

    #lfdf = RESTORED_DEMAND.groupby(group_cols).apply(
    #    lambda x : x.set_index('PERIOD_DT').rolling(window=str(DR_PARAMETERS['DR_PERIOD_LENGTH']) + 'd').agg('sum')
    #)['DEFICIT_FLG1']
    #lfdf = lfdf.reset_index().rename(columns={'DEFICIT_FLG1' : 'COUNT_NONDEFICIT_DAYS'})
    #RESTORED_DEMAND = pd.merge(RESTORED_DEMAND, lfdf.reset_index(), 'left', on=group_cols + ['PERIOD_DT'])
    #RESTORED_DEMAND['DEFICIT_FLG1'] = 1 - RESTORED_DEMAND['DEFICIT_FLG1']
    #RESTORED_DEMAND = RESTORED_DEMAND.drop('index', axis=1)
    RESTORED_DEMAND['COUNT_NONDEFICIT_DAYS'] = np.nan
    RESTORED_DEMAND['MEAN'] = np.nan
    RESTORED_DEMAND['STD'] = np.nan

    STOCK['PERIOD_DT'] = np.random.choice(FORECAST_FLAG['PERIOD_START_DT'], size=STOCK.shape[0])
    STOCK = STOCK.drop_duplicates(subset=['PRODUCT_ID', 'LOCATION_ID', 'PERIOD_DT'])

    PROMO['PERIOD_START_DT'] = FORECAST_FLAG['PERIOD_START_DT'][0]
    PROMO['PERIOD_END_DT'] = FORECAST_FLAG['PERIOD_START_DT'][0] + datetime.timedelta(days=30)
    
    PRODUCT_ATTR.iloc[0, 1:] = ['EA', 1]
    PRODUCT_ATTR
    return SALES, FORECAST_FLAG, RESTORED_DEMAND, PROMO, STOCK, PRODUCT_ATTR


def prepare_sales_and_demand(FORECAST_FLAG : pd.DataFrame, DR_PARAMETERS : dict, IB_HIST_END_DT : datetime.datetime,
                             IB_UPDATE_HISTORY_DEPTH : int, SALES : pd.DataFrame) -> pd.DataFrame:
    """
    Step 3.1
    
    Function preparating sales
    
    Parameters
    ----------
    FORECAST_FLAG : pd.DataFrame
        FORECAST_FLAG table
    DR_PARAMETERS : dict
        Configuration parameters used within the step
    IB_HIST_START_DT : datetime.datetime
        Minimal date that should be present in Demand Restored
    IB_UPDATE_HISTORY_DEPTH : int
        Number of days of historical information that should be considered within this step running,
        i.e. only dates since (>=) IB_HIST_END_DT -  IB_UPDATE_HISTORY_DEPTH should be used within step
    SALES
        Sales table
    
    Returns
    -------
    pd.DataFrame
        T1 table
    
    """
    df = FORECAST_FLAG[FORECAST_FLAG['STATUS'] == 'maturity'] ## Step 3.1.2 active == maturity ???
    ## Step 3.1.3
    df['PERIOD_END_DT'] = df['PERIOD_END_DT'] + datetime.timedelta(days=DR_PARAMETERS['DR_LIFECYCLE_MARGIN'])
    df.loc[df['PERIOD_END_DT'] > pd.to_datetime(
        IB_HIST_END_DT.strftime("%Y-%m-%d")), 'PERIOD_END_DT'] = pd.to_datetime(IB_HIST_END_DT.strftime("%Y-%m-%d")
    )

    ## Step 3.1.4

    moment = pd.to_datetime((IB_HIST_END_DT - datetime.timedelta(days=IB_UPDATE_HISTORY_DEPTH)).strftime("%Y-%m-%d"))
    moment2 = pd.to_datetime((
        IB_HIST_END_DT - datetime.timedelta(days=IB_UPDATE_HISTORY_DEPTH) - datetime.timedelta(
            days=DR_PARAMETERS['DR_PERIOD_LENGTH'])).strftime("%Y-%m-%d")
    )
    df[(df['PERIOD_START_DT'] >= moment) | (df['PERIOD_START_DT'] >= moment2) |
                 (df['PERIOD_END_DT'] >= moment2)]

    if IB_UPDATE_HISTORY_DEPTH <= 0:
        moment = (IB_HIST_START_DT - datetime.timedelta(DR_PARAMETERS['DR_PERIOD_LENGTH']))
        df.loc[df['PERIOD_START_DT'] <= moment, 'PERIOD_START_DT'] = moment
    else:
        moment = IB_HIST_END_DT - datetime.timedelta(days=max(IB_UPDATE_HISTORY_DEPTH, 0)) - datetime.timedelta(days=DR_PARAMETERS['DR_PERIOD_LENGTH'])
        df.loc[df['PERIOD_START_DT'] < moment, 'PERIOD_START_DT'] = moment
        df.loc[df['PERIOD_END_DT'] > IB_HIST_END_DT, 'PERIOD_END_DT'] = IB_HIST_END_DT

    ## Step 3.1.5


    keys = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'PERIOD_DT']

    T1 = pd.merge(df.rename(columns={'PERIOD_START_DT' : 'PERIOD_DT'}), SALES, how='left', on=keys).rename(
        columns={'SALES_QTY' : 'TGT_QTY'})
    
    return T1


def add_stock_data_and_promo_flag(T1 : pd.DataFrame, STOCK : pd.DataFrame, PROMO : pd.DataFrame) -> pd.DataFrame:
    """
    Steps 3.2 and 3.3
    Function adding the stock data and promo flag
    
    Parameters
    ----------
    T1 : pd.DataFrame
        T1 table received in the previous step
    STOCK : pd.DataFrame
        STOCK table
    PROMO : pd.DataFrame
        PROMO table
    
    Returns
    -------
    pd.DataFrame
        T3 table
    """
    STOCK['PERIOD_DT'] = pd.to_datetime(STOCK['PERIOD_DT'])
    keys = ['PRODUCT_ID', 'LOCATION_ID', 'PERIOD_DT']
    T2 = pd.merge(T1, STOCK[keys + ['STOCK_QTY']], on=keys)

    ## Step 3.3

    keys = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID', 'PROMO_ID']
    T3 = T2[T2['PROMO_ID'].notna()]
    T3['PROMO_ID'] = T3['PROMO_ID'].astype(int)
    T3 = pd.merge(T3, PROMO[keys + ['PERIOD_START_DT', 'PERIOD_END_DT', 'PROMO_PRICE']], on=keys, how='left', suffixes=['', '_promo'])
    T3 = T3[(T3['PERIOD_DT'] >= T3['PERIOD_START_DT']) & (T3['PERIOD_DT'] <= T3['PERIOD_END_DT_promo'])]
    keys = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']
    group = T3.groupby(keys).mean()[['TGT_QTY', 'STOCK_QTY']].reset_index()
    T3 = pd.merge(T3, group, on=keys, suffixes=['_x', ''])
    group = T3.groupby(keys).max()[['PROMO_FLG', 'PROMO_ID']].reset_index()
    T3 = pd.merge(T3, group, on=keys, suffixes=['_x', ''])
    T3 = T3.drop(['TGT_QTY_x', 'STOCK_QTY_x', 'PROMO_FLG_x', 'PROMO_ID_x'], axis=1)
    keys = T2.columns.tolist()
    T3 = pd.merge(T2, T3, how='left', on=keys)
    return T3


def primiry_deficit_flg_def(T3 : pd.DataFrame, DR_PARAMETERS : dict) -> pd.DataFrame:
    """
    Step 3.4.1
    Function adding primiry deficit flag
    
    Parameters
    ----------
    T3 : pd.DataFrame
        T3 table received in the previous step
    DR_PARAMETERS : dict
        Configuration parameters used within the step
    
    Returns
    -------
    pd.DataFrame
        T41 table
    """
    keys = ['LOCATION_ID', 'PERIOD_DT']
    group = T3.groupby(keys)[['TGT_QTY']].sum().rename(columns={'TGT_QTY' : 'SUM_TGT_QTY'})
    T3 = pd.merge(T3, group, on=keys)
    T3['CLOSED_FLG'] = 1 - (T3['SUM_TGT_QTY'] < DR_PARAMETERS['MIN_SALES_QTY_DAY']).astype(int)
    T3['DEFICIT_FLG1'] = ((T3['STOCK_QTY'] < DR_PARAMETERS['DEF_INV_TRSHD']) & (T3['TGT_QTY'] < DR_PARAMETERS['DEF_QTY_TRSHD'])).astype(int)
    T41 = T3
    return T41



def secondary_deficit_flg_def(T41 : pd.DataFrame, IB_HIST_START_DT : datetime.datetime, IB_HIST_END_DT : datetime.datetime,
                             IB_UPDATE_HISTORY_DEPTH : int, HIGH_TURNOVER_TRSHD : float, DR_PARAMETERS : dict):
    """
    Step 3.4.2
    Function adding secondary deficit flag and calculating mean std and count
    
    Parameters
    ----------
    T41 : pd.DataFrame
        T41 table received in the previous step
    IB_HIST_START_DT : datetime.datetime
        Minimal date that should be present in Demand Restored
    IB_HIST_END_DT : datetime.datetime
        Last known date (i.e. sales and stock information is known)
    IB_UPDATE_HISTORY_DEPTH : int
        Number of days of historical information that should be considered within this step running,
        i.e. only dates since (>=) IB_HIST_END_DT -  IB_UPDATE_HISTORY_DEPTH should be used within step
    HIGH_TURNOVER_TRSHD : float
        Parameter
    DR_PARAMETERS : dict
        Configuration parameters used within the step
        
    Returns
    -------
    pd.DataFrame
        T42 table
    """
    if IB_UPDATE_HISTORY_DEPTH <= 0:
        d = IB_HIST_START_DT
    else:
        d = IB_HIST_END_DT - datetime.timedelta(IB_UPDATE_HISTORY_DEPTH)

    T42 = T41[T41['PERIOD_DT'] >= d]
    T42['DEFICIT_FLG1'] = 1 - T42['DEFICIT_FLG1']
    keys = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']
    lfdf = T42.groupby(keys).apply(
        lambda x : x.set_index('PERIOD_DT')[['TGT_QTY', 'DEFICIT_FLG1']].rolling(
            window=str(DR_PARAMETERS['DR_PERIOD_LENGTH']) + 'd').agg(
            {'TGT_QTY' : ['mean', 'std'], 'DEFICIT_FLG1' : 'count'})
    )
    T42['DEFICIT_FLG1'] = 1 - T42['DEFICIT_FLG1']
    lfdf.columns = ['mean', 'std', 'COUNT_NONDEFECIT_DAYS']
    T42 = pd.merge(T42, lfdf, on=keys)
    T42 = T42.ffill()
    T42['mean'] = T42['mean'].fillna(0)
    T42['std'] = T42['std'].fillna(1)
    T42['Threshold'] = T42['mean'] - 2 * T42['std']
    T42['DEFICIT_FLG2'] = 0
    T42.loc[T42['DEFICIT_FLG1'] == 1, 'DEFICIT_FLG2'] = 1
    T42.loc[T42['TGT_QTY'] < T42['Threshold'], 'DEFICIT_FLG2'] = 1
    T42.loc[(T42['mean'] >= HIGH_TURNOVER_TRSHD) & (T42['TGT_QTY'] < T42['mean'] * 0.1), 'DEFICIT_FLG2'] = 1
    return T42


def demand_restoration_on_stock_def(T42 : pd.DataFrame, DR_PARAMETERS : dict) -> pd.DataFrame:
    """
    Step 3.4.3
    Function restoring demand basing on stock deficit
    
    Parameters
    ----------
    T42 : pd.DataFrame
        T42 table received in the previous step
    DR_PARAMETERS : dict
        Configuration parameters used within the step
    
    Returns
    -------
    pd.DataFrame
        T43 table
    """
    T42['TGT_QTY_R'] = T42['TGT_QTY']
    T42.loc[(T42['DEFICIT_FLG2'] == 1) & (T42['COUNT_NONDEFECIT_DAYS'] >= DR_PARAMETERS['MIN_ND_DAYS']),
            'TGT_QTY_R'] = T42[(T42['DEFICIT_FLG2'] == 1) & (T42['COUNT_NONDEFECIT_DAYS'] >= DR_PARAMETERS['MIN_ND_DAYS'])]['mean']
    T43 = T42
    return T43


def history_extending(T43 : pd.DataFrame, PRODUCT_ATTR : pd.DataFrame, SEASONAL_FLAG_CONFIG : pd.DataFrame,
                     DR_PARAMETERS : dict, IB_HIST_END_DT : datetime.datetime) -> pd.DataFrame:
    """
    Step 3.4.5
    Function extending demand series for short and seasonal time series
    
    Parameters
    ----------
    T43 : pd.DataFrame
        T43 table received in the previous step
    PRODUCT_ATTR : pd.DataFrame
        PRODUCT_ATTR table
    SEASONAL_FLAG_CONFIG : pd.DataFrame
        SEASONAL_FLAG_CONFIG table
    DR_PARAMETERS : dict
        Configuration parameters used within the step
    IB_HIST_END_DT : datetime.datetime
        Last known date (i.e. sales and stock information is known)
        
    Returns
    -------
    pd.DataFrame
        Restored demand table
    
    """
    products = pd.merge(PRODUCT_ATTR, SEASONAL_FLAG_CONFIG, on=['PRODUCT_ATTR_NAME', 'PRODUCT_ATTR_VALUE'])['PRODUCT_ID'].tolist()
    df = T43[T43['PRODUCT_ID'].isin(products)]
    keys = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID']
    df['intnx1'] = df['PERIOD_DT'] - np.timedelta64(1, "M")
    df['intnx2'] = df['PERIOD_END_DT']
    df.loc[df['intnx2'] > IB_HIST_END_DT, 'intnx2'] = IB_HIST_END_DT
    df['intnx2'] = df['intnx2'] - np.timedelta64(1, "M")
    group = df.groupby(keys).agg({'intnx1' : 'min', 'intnx2' : 'max'})
    group['HISTORY_DEPTH'] = ((group['intnx2'] - group['intnx1']) / np.timedelta64(1, 'M')).astype(int)
    group = group.rename(columns={'intnx1' : 'MIN_MONTH_DT', 'intnx2' : 'MAX_MONTH_DT'})
    df = pd.merge(T43, group.reset_index(), on=keys)
    df = df[(df['HISTORY_DEPTH'] >= DR_PARAMETERS['MIN_PROLONG_HIST_MONTH']) &
       (df['HISTORY_DEPTH'] <= DR_PARAMETERS['MAX_PROLONG_HIST_MONTH']) &
           (abs(df['MAX_MONTH_DT'] - datetime.datetime.today()) / np.timedelta64(1, 'M') < 12)]
    medians = df.groupby(keys)[['TGT_QTY_R']].median().reset_index()
    calendar = pd.DataFrame()
    today = datetime.datetime.today()
    calendar['PERIOD_DT'] = (pd.date_range(today.replace(year=today.year - 2), today, freq='M') + datetime.timedelta(days=1))
    calendar['PERIOD_DT'] = pd.to_datetime(calendar['PERIOD_DT']).dt.date
    for key in keys:
        calendar = pd.merge(calendar, medians[[key]], how='cross')
    calendar = pd.merge(calendar, medians, on=keys)

    df = pd.concat([T43[~T43['PRODUCT_ID'].isin(products)], calendar])
    return df


def demand_restoration_algorithm(DR_PARAMETERS : dict, TGT_VAR_CONFIG : dict,
                  IB_MAX_DT : datetime.datetime, IB_UPDATE_HISTORY_DEPTH : int,
                 IB_HIST_START_DT : datetime.datetime, IB_HIST_END_DT : datetime.datetime,
                 hierarchies : dict, STOCK : pd.DataFrame, PROMO : pd.DataFrame, 
                SALES : pd.DataFrame, FORECAST_FLAG : pd.DataFrame,
                RESTORED_DEMAND : pd.DataFrame, HIGH_TURNOVER_TRSHD : float,
                PRODUCT_ATTR : pd.DataFrame, SEASONAL_FLAG_CONFIG : pd.DataFrame) -> pd.DataFrame:
    """
    This step defines how sales history should be treated as a demand.
    Particularly, the main purpose of this transformation is to provide downstream steps
        with as most as possible correct information regarding unconstrained demand observed in the past.
        
    Parameters
    ----------
    DR_PARAMETERS : dict
        Configuration parameters used within the step
    TGT_VAR_CONFIG : dict
        Configuration parameters used within the step
    IB_MAX_DT
        Maximal date which is used for data preparation (e.g. 01/01/2100)
    IB_UPDATE_HISTORY_DEPTH : int
        Number of days of historical information that should be considered within this step running,
        i.e. only dates since (>=) IB_HIST_END_DT -  IB_UPDATE_HISTORY_DEPTH should be used within step
    IB_HIST_START_DT : datetime.datetime
        Minimal date that should be present in Demand Restored
    IB_HIST_END_DT : datetime.datetime
        Last known date (i.e. sales and stock information is known)
    hierarchies : dict
        Dictionary containg matches of key names with the relevant hierarchical tables
    STOCK : pd.DataFrame
        STOCK table
    PROMO : pd.DataFrame
        PROMO table
    SALES : pd.DataFrame
        SALES table
    FORECAST_FLAG : pd.DataFrame
        FORECAST_FLAG table
    RESTORED_DEMAND
        RESTORED_DEMAND table
    HIGH_TURNOVER_TRSHD
        Parameter
    PRODUCT_ATTR : pd.DataFrame
        PRODUCT_ATTR table
    SEASONAL_FLAG_CONFIG : pd.DataFrame
        SEASONAL_FLAG_CONFIG table
        
    Returns
    -------
    pd.DataFrame
        Table containing restored demand
    """
    T1 = prepare_sales_and_demand(FORECAST_FLAG, DR_PARAMETERS, IB_HIST_END_DT, IB_UPDATE_HISTORY_DEPTH, SALES)
    T3 = add_stock_data_and_promo_flag(T1, STOCK, PROMO)
    T41 = primiry_deficit_flg_def(T3, DR_PARAMETERS)
    T42 = secondary_deficit_flg_def(T41, IB_HIST_START_DT, IB_HIST_END_DT, IB_UPDATE_HISTORY_DEPTH, HIGH_TURNOVER_TRSHD, DR_PARAMETERS)
    T43 = demand_restoration_on_stock_def(T42, DR_PARAMETERS)
    df = history_extending(T43, PRODUCT_ATTR, SEASONAL_FLAG_CONFIG, DR_PARAMETERS, IB_HIST_END_DT)
    df['PERIOD_DT'] = pd.to_datetime(df['PERIOD_DT']).dt.date
    return df