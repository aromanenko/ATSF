import numpy as np, pandas as pd
from datetime import date
import datetime

FORECAST_FLAG = None

IB_UPDATE_HISTORY_DEPTH = None
IB_MAX_DT = None
IB_HIST_END_DT = None

def generate_input_unfolding(size):
  """
  Function to generate input data for unfolding algorithm

  Parameters
  ----------
  size : int
    Number of rows in generated tables

  Returns
  -------
    pd.DataFrame
  Table to be unfolded later

    pd.DataFrame
  Product Hierarchy
  
    pd.DataFrame
  Location Hierarchy

    pd.DataFrame
  Customer Hierarchy

    pd.DataFrame
  Distribution Channel Hierarchy

  """
  mplnTable = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(4)],axis=1)
  mplnTable.columns = ["product_lvl_id", "location_lvl_id", "customer_lvl_id", "distr_channel_lvl_id"]

  mpInProduct = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(26)],axis=1)
  mpInProduct = pd.concat([mpInProduct, pd.Series(np.random.choice([0,1],size))],axis=1)
  mpInProduct.columns = ['product_lvl_id', 'PRODUCT_LVL_NM1', 'PRODUCT_LVL_DESC1',
       'PRODUCT_LVL_ID2', 'PRODUCT_LVL_NM2', 'PRODUCT_LVL_DESC2',
       'PRODUCT_LVL_ID3', 'PRODUCT_LVL_NM3', 'PRODUCT_LVL_DESC3',
       'PRODUCT_LVL_ID4', 'PRODUCT_LVL_NM4', 'PRODUCT_LVL_DESC4',
       'PRODUCT_LVL_ID5', 'PRODUCT_LVL_NM5', 'PRODUCT_LVL_DESC5',
       'PRODUCT_LVL_ID6', 'PRODUCT_LVL_NM6', 'PRODUCT_LVL_DESC6',
       'PRODUCT_LVL_ID7', 'PRODUCT_LVL_NM7', 'PRODUCT_LVL_DESC7',
       'parent_product_id', 'PRODUCT_ID', 'PRODUCT_NM', 'PRODUCT_DESC',
       'MODIFIED_DTTM', 'DELETE_FLG']
       
  mpInLocation = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(21)],axis=1)
  mpInLocation = pd.concat([mpInLocation, pd.Series(np.random.choice([0,1],size))],axis=1)
  mpInLocation.columns = ['location_lvl_id', 'location_lvl_nm1', 'location_lvl_desc1',
       'location_lvl_id2', 'location_lvl_nm2', 'location_lvl_desc2',
       'location_lvl_id3', 'location_lvl_nm3', 'location_lvl_desc3',
       'location_lvl_id4', 'location_lvl_nm4', 'location_lvl_desc4',
       'location_lvl_id5', 'location_lvl_nm5', 'location_lvl_desc5',
       'location_id', 'location_nm', 'location_desc', 'open_dttm',
       'close_dttm', 'modified_dttm','del_flag']

  mpInCustomer = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(8)],axis=1)
  mpInCustomer = pd.concat([mpInCustomer, pd.Series(np.random.choice([0,1],size))],axis=1)
  mpInCustomer.columns = ['customer_lvl_id','location_lvl_desc5',
       'location_id', 'location_nm', 'location_desc', 'open_dttm',
       'close_dttm', 'modified_dttm','del_flag']

  mpInDistrChannel = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(8)],axis=1)
  mpInDistrChannel = pd.concat([mpInDistrChannel, pd.Series(np.random.choice([0,1],size))],axis=1)
  mpInDistrChannel.columns = ['distr_channel_lvl_id','location_lvl_desc5',
       'location_id', 'location_nm', 'location_desc', 'open_dttm',
       'close_dttm', 'modified_dttm','del_flag']
  

  return mplnTable,mpInProduct,mpInLocation,mpInCustomer,mpInDistrChannel



def unfold_aggregated_data(mpInTable, mpInProduct, mpInLocation, mpInCustomer,mpInDistrChannel, mpInQuadruple = None):
  """
  Utility to unfold aggregated data to lower level of organizational hierarchy (3)

  Parameters
  ----------
  mpInTable : pd.DataFrame
    Input Table that must be unfold

  mpInProduct : pd.DataFrame
    Product Hierarchy

  mpInLocation : pd.DataFrame
    Location Hierarchy

  mpInCustomer : pd.DataFrame
    Customer Hierarchy

  mpInDistrChannel : pd.DataFrame
    Distribution channel Hierarchy

  Returns
  -------
  pd.DataFrame
    Returns unfolded table
  """

  table = mpInTable.merge(mpInProduct, on=["product_lvl_id"], how="left")
  table = table.merge(mpInLocation, on=["location_lvl_id"], how="left")
  table = table.merge(mpInCustomer, on=["customer_lvl_id"], how="left")
  table = table.merge(mpInDistrChannel, on=["distr_channel_lvl_id"], how="left")
  if mpInQuadruple is not None:  
    table = table.merge(mpInQuadruple, on=['product_id', 'location_id', 'customer_id', 'distr_channel_id'], how = 'inner')

  return table



def generate_input_ilp(start_date, end_date):
  """
  Function to generate input data for incremental load preparation

  Parameters
  ----------
  start_date : datetime.datetime
    First date of sales

  end_date : datetime.datetime
  	Last date of sales

  Returns
  -------
    pd.DataFrame
  Sales Table

    pd.DataFrame
  Stocks Hierarchy
  
    pd.DataFrame
  Sell in  Hierarchy

    pd.DataFrame
  Sell out Hierarchy

    pd.DataFrame
  Assortment Matrix 

  	pd.DataFrame
  Location File Hierarchy

  """

  IB_UPDATE_HISTORY_DEPTH = 3
  IB_MAX_DT = datetime.date.fromisoformat('2030-12-12')
  IB_HIST_END_DT = datetime.date.fromisoformat('2003-12-01')

  date = pd.date_range(start_date, end_date, freq='D')
  size = len(date)

  sales = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(5)],axis=1)
  sales = pd.concat([pd.Series(date),sales],axis=1)
  #sales = pd.concat([sales,pd.Series(np.full(0,size))])
  sales.columns = ['period_dt','product_id', 'location_id', 'customer_id','distr_channel_id','period_start_dt']

  stock = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(3)],axis=1)
  stock = pd.concat([stock, pd.Series(date)],axis=1)
  stock.columns = ['PRODUCT_ID','LOCATION_ID','STOCK_QTY','PERIOD_START_DT']

  

  sell_in = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(16)],axis=1)
  sell_in = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(16)],axis=1)
  sell_in = pd.concat([sell_in, pd.Series(date)],axis=1)
  sell_in = pd.concat([sell_in, pd.Series(np.random.choice([0,1],size))],axis=1)
  sell_in.columns = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID',
       'PERIOD_DT', 'ORDERS_QTY', 'ORDERS_AMOUNT', 'SHIPMENTS_QTY',
       'SHIPMENTS_AMOUNT', 'INVOICES_QTY', 'INVOICES_AMOUNT', 'RETURNS_QTY',
       'RETUNRS_AMOUNT', 'PROMO_FLG', 'PROMO_ID', 'COST', 'date', 'del_flag']

  sell_out = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(16)],axis=1)
  sell_out = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(16)],axis=1)
  sell_out = pd.concat([sell_out, pd.Series(date)],axis=1)
  sell_out = pd.concat([sell_out, pd.Series(np.random.choice([0,1],size))],axis=1)
  sell_out.columns = ['PRODUCT_ID', 'LOCATION_ID', 'CUSTOMER_ID', 'DISTR_CHANNEL_ID',
       'PERIOD_DT', 'ORDERS_QTY', 'ORDERS_AMOUNT', 'SHIPMENTS_QTY',
       'SHIPMENTS_AMOUNT', 'INVOICES_QTY', 'INVOICES_AMOUNT', 'RETURNS_QTY',
       'RETUNRS_AMOUNT', 'PROMO_FLG', 'PROMO_ID', 'COST', 'date', 'del_flag']


  ASSORT_MATRIX = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(4)],axis=1)
  ASSORT_MATRIX = pd.concat([ASSORT_MATRIX, pd.Series(date)],axis=1)
  ASSORT_MATRIX = pd.concat([ASSORT_MATRIX, pd.Series(date+datetime.timedelta(days=15))],axis=1)
  ASSORT_MATRIX = pd.concat([ASSORT_MATRIX, pd.Series(np.random.choice([0,1],size))],axis=1)
  ASSORT_MATRIX.columns = ["product_id", "location_id", "customer_id", "distr_channel_id", "stard_td",
  	"end_td", 'del_flag']


  LOCATION_LIFE = pd.concat([pd.Series(np.arange(10000,10000+size)) for _ in range(6)],axis=1)
  LOCATION_LIFE = pd.concat([LOCATION_LIFE, pd.Series(date)],axis=1)
  LOCATION_LIFE.columns = ['product_id', 'location_id', 'customer_id', 'distr_channel_id',
       'PERIOD_DT', 'ORDERS_QTY', 'date']
  LOCATION_LIFE.PERIOD_TYPE = pd.Series(np.random.choice(['reconstruction','re-branding'],size))
  LOCATION_LIFE.del_flag = pd.Series(np.random.choice([0,1],size))


  return sales, stock, sell_in, sell_out, ASSORT_MATRIX,LOCATION_LIFE

def time_interval_utility(table, granuality, distance_tolerance,groupby):
  """
	Technical utility for time-intervals union

	Parameters
  	----------
  	Sales : pd.DataFrame
    	Input table to transform 

    granuality : str 
    	Granuality to put dates in accordance with

    groubpy : str 
    	Column to group by

   	Returns
   	----------
   	list
   		transformed dates list
	"""
  
  dates = []
  
  for i in range(len(table.groupby(groupby))):
      if len(list(table.groupby(groupby))[i][1]):
          dates.append(table.groupby(groupby))[i][1]
      else: 
    	    grouped_by = list(table.groupby(groupby))
    	    for j in range(1,len(list(table.groupby(groupby))[i][1])-1):
      	      left_bound = grouped_by[j][1].start_dt
      	      right_bound = grouped_by[j][1].end_dt
      	      next_left_bound = grouped_by[j+1][1].start_dt
      	      next_right_bound = grouped_by[j+1][1].end_dt

      	      if (left_bound < next_right_bound and next_left_bound > next_left_bound \
                 and right_bound > next_right_bound):
                  grouped_by[j+1][1].end_dt = right_bound
			
      dates.append((right_bound,left_bound))
      return dates


def incremental_load_preparation(SALES, STOCK, SELL_IN, SELL_OUT, ASSORT_MATRIX, LOCATION_LIFE, start_date, end_date, PRODUCT_LIFE = None, CUSTOMER_LIFE = None):
  """
  Utility to unfold aggregated data to lower level of organizational hierarchy (3.1)

  Parameters
  ----------
  Sales : pd.DataFrame
    Sales Table

  Stock : pd.DataFrame
    Stocks Table

  SELL_IN : pd.DataFrame
    Sells in Table

  SELL_OUT : pd.DataFrame
    Sells out Table

  ASSORT_MATRIX : pd.DataFrame
    Assortment matrix

  Returns
  -------
  pd.DataFrame
    Qudrabples to delete

  pd.DataFrame
    Updated assortment matrix

  """

  if FORECAST_FLAG == None:
    IB_UPDATE_HISTORY_DEPTH = 0

  IB_UPDATE_HISTORY_DEPTH = 3
  IB_MAX_DT = date.fromisoformat('2030-12-12')
  IB_HIST_END_DT = date.fromisoformat('2003-12-01')

  table1 = SALES[['product_id', 'location_id', 'customer_id', 'distr_channel_id','period_dt']].drop_duplicates()
  table2 = ASSORT_MATRIX[['product_id', 'location_id', 'customer_id',\
                          'distr_channel_id']][ASSORT_MATRIX.del_flag == 1].drop_duplicates()
  table3 = LOCATION_LIFE[['product_id', 'location_id', 'customer_id',\
                          'distr_channel_id']][LOCATION_LIFE.del_flag == 1].drop_duplicates()

  QUADRUPLES_DELETE = pd.concat([table1,table2,table2],axis=0)

  SALES_UPDATE_FF = SALES.merge(QUADRUPLES_DELETE, on = ['product_id', 'location_id'],how='left')
  SALES_UPDATE_FF.IB_UPDATE_HISTORY_DEPTH = pd.Series(np.full(shape = len(SALES_UPDATE_FF), fill_value=IB_UPDATE_HISTORY_DEPTH))

  SALES_UPDATE_FF = SALES_UPDATE_FF.loc[(SALES_UPDATE_FF.IB_UPDATE_HISTORY_DEPTH<=0) | (pd.to_datetime(SALES_UPDATE_FF.period_dt_x).dt.date > (IB_HIST_END_DT - datetime.timedelta(days=IB_UPDATE_HISTORY_DEPTH))) | (SALES_UPDATE_FF.product_id is not np.nan)]
  
  ASSORT_MATRIX_UPDATE_FF = ASSORT_MATRIX.merge(QUADRUPLES_DELETE, on = ['product_id', 'location_id', 'customer_id', \
                          'distr_channel_id'],how='left')
  ASSORT_MATRIX_UPDATE_FF.IB_UPDATE_HISTORY_DEPTH = pd.Series(np.full(shape = len(ASSORT_MATRIX_UPDATE_FF), fill_value=IB_UPDATE_HISTORY_DEPTH))
  ASSORT_MATRIX_UPDATE_FF = ASSORT_MATRIX_UPDATE_FF.loc[(ASSORT_MATRIX_UPDATE_FF.IB_UPDATE_HISTORY_DEPTH<=0) | (pd.to_datetime(ASSORT_MATRIX_UPDATE_FF.period_dt).dt.date > (IB_HIST_END_DT - datetime.timedelta(days=IB_UPDATE_HISTORY_DEPTH))) | (ASSORT_MATRIX_UPDATE_FF.product_id is not np.nan)]

  ASSORT_MATRIX_UPDATE_FF.IB_UPDATE_HISTORY_DEPTH = pd.Series(np.full(shape = len(SALES_UPDATE_FF), fill_value=0))
  ASSORT_MATRIX_UPDATE_FF.period_dt = pd.Series(pd.date_range(start_date, end_date, freq='D'))
  ASSORT_MATRIX_UPDATE_FF.IB_UPDATE_HISTORY_DEPTH_date = pd.Series(pd.date_range(start_date, end_date, freq='D'))
  ASSORT_MATRIX_UPDATE_FF = ASSORT_MATRIX_UPDATE_FF.loc[(ASSORT_MATRIX_UPDATE_FF.IB_UPDATE_HISTORY_DEPTH<=0)]
  
  return SALES_UPDATE_FF, ASSORT_MATRIX_UPDATE_FF,QUADRUPLES_DELETE


def adding_fields(sales, stock, assort):
  """
  Function to add required fields into sales and stock (3.2)

  Parameters
  ----------
  Sales : pd.DataFrame
    Sales Table

  Stock : pd.DataFrame
    Stocks Table

  Assort : pd.DataFrame
    Assortment matrix

  Returns
  -------
  pd.DataFrame
    L1  table

  """
  table1 = sales[['location_id','product_id','period_dt_y']]
  table1.columns = ['location_id','product_id','period_dt']
  table2 = stock[['LOCATION_ID','PRODUCT_ID','PERIOD_START_DT']]
  table2.columns = ['location_id','product_id','period_dt']

  res = pd.concat([table1,table2],axis=0)

  res = res.merge(assort, on=['product_id','location_id'],how='left')

  res.customer = assort.customer_id
  res.distr_channel_id = assort.distr_channel_id

  print(res.columns)
  res.customer.fillna(method='backfill',inplace=True)
  res.distr_channel_id.fillna(method='bfill',inplace=True)

  res.customer.fillna(res.customer.min())
  res.distr_channel_id.fillna(res.distr_channel_id.min())
  res.drop(['period_dt_x','period_dt_y'],axis=1,inplace=True) # drop these columns, as they appear because of data-generating functions.
  
  return res
