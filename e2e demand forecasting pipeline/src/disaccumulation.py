import pandas as pd
from tqdm.notebook import tqdm

import os
import sys

project_path = os.path.abspath(os.path.join('..'))

if project_path not in sys.path:
    sys.path.append(project_path)


class Disaccumulation:
    def __init__(self, data, out_time_lvl):
        """
        Provide forecasts at the required time granularity level.
        
        Parameters
        ----------
        data : pd.DataFrame
            Table with ID, Period and Forecast columns
        
        out_time_lvl : string
            Required time granularity level
            
            Possible values:
            
            D - days
            W - weeks (default starting from Sunday)
            W-MON/W-TUE/.../W-SUN - weeks, starting from specified day of week
            M - months
        """
        self.data = data
        self.data_splitted = data
        self.out_time_lvl = out_time_lvl
        self.FINAL_GRANULARITY_DELIVERED = True
        
        
    def check_granulatiry(self):
        """
        Check whether period_dt and period_end_dt in AGG_HYB_FCST correspond to out_time_lvl
        
        Returns
        -------
        bool
            Returns flag which shows whether forecast split needed or not
        """
        if self.out_time_lvl == 'D':
            if (self.data['PERIOD_DT'] != self.data['PERIOD_END_DT']).any():
                self.FINAL_GRANULARITY_DELIVERED = False
        elif 'W' in self.out_time_lvl:
            if (self.data['PERIOD_DT'].apply(lambda x: x - pd.Timedelta(f'{x.dayofweek}D')) != \
                self.data['PERIOD_END_DT'].apply(lambda x: x - pd.Timedelta(f'{x.dayofweek}D'))).any():
                self.FINAL_GRANULARITY_DELIVERED = False
        if self.out_time_lvl == 'M':
            if (self.data['PERIOD_DT'].apply(lambda x: x - pd.Timedelta(f'{x.day - 1}D')) != \
                self.data['PERIOD_END_DT'].apply(lambda x: x - pd.Timedelta(f'{x.day - 1}D'))).any():
                self.FINAL_GRANULARITY_DELIVERED = False
                
        return self.FINAL_GRANULARITY_DELIVERED
            
        
    def change_granularity(self):
        """
        If FINAL_GRANULARITY_DELIVERED == False then transform original table by splitting forecast periods
        to more granular time stamps
        
        Returns
        -------
        pd.DataFrame
            Splitted data to more granular time stamps
        """
        df = self.data.copy()
        df['OUT_PERIOD_DT'] = df['PERIOD_DT']
        df['OUT_PERIOD_END_DT'] = df['PERIOD_END_DT']

        for ind, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            cur_dates = pd.to_datetime(np.array([row['PERIOD_DT'], row['PERIOD_END_DT']]))
            split_dates = pd.period_range(cur_dates[0], cur_dates[1], freq=self.out_time_lvl).to_timestamp()
            taken_dates = split_dates[(split_dates > cur_dates[0]) & (split_dates < cur_dates[1])]
            needed_dates = np.append(taken_dates, cur_dates)
            needed_dates.sort()
            start_ind = 1
            if self.out_time_lvl == 'D':
                start_ind = 0
                df.loc[ind, 'OUT_PERIOD_END_DT'] = needed_dates[1] - pd.Timedelta('1D')
            else:
                df.loc[ind, 'OUT_PERIOD_END_DT'] = needed_dates[1]

            for i in range(start_ind, len(needed_dates) - 1):        
                new_row = row.copy()
                new_row['OUT_PERIOD_DT'] = needed_dates[i] + pd.Timedelta('1D')
                new_row['OUT_PERIOD_END_DT'] = needed_dates[i + 1]
                df = df.append(new_row, ignore_index=True)

        self.data_filled = df.sort_values(df.columns[df.columns.str.contains('_ID')].to_list() + ['PERIOD_DT']).reset_index(drop=True)
        
        return self.data_filled
    
    def share_forecast(self):
        """
        Calculate forecast share and volume of VF_FORECAST_VALUE, ML_FORECAST_VALUE,
        HYBRID_FORECAST proportionally to number of days in interval [PERIOD_DT, PERIOD_END_DT]
        
        Returns
        -------
        pd.DataFrame
            Data with shared forecast
        """
        def split(x, target):
            return x[target] * ((x['OUT_PERIOD_END_DT'] - x['OUT_PERIOD_DT']) / np.timedelta64(1, 'D') + 1) / \
        ((x['PERIOD_END_DT'] - x['PERIOD_DT']) / np.timedelta64(1, 'D') + 1)

        self.data_filled['VF_FORECAST_VALUE'] = self.data_filled.apply(lambda x: split(x, 'VF_FORECAST_VALUE'), axis=1)
        self.data_filled['ML_FORECAST_VALUE'] = self.data_filled.apply(lambda x: split(x, 'ML_FORECAST_VALUE'), axis=1)
        self.data_filled['HYBRID_FORECAST_VALUE'] = self.data_filled.apply(lambda x: split(x, 'HYBRID_FORECAST_VALUE'), axis=1)

        self.data_filled = self.data_filled.drop(['PERIOD_DT', 'PERIOD_END_DT'], axis=1)
        self.data_filled = self.data_filled.rename(columns={'OUT_PERIOD_DT': 'PERIOD_DT', 'OUT_PERIOD_END_DT': 'PERIOD_END_DT'})
        self.data_filled = self.data_filled.set_index(['PERIOD_DT', 'PERIOD_END_DT']).reset_index()
        self.data_splitted = self.data_filled
        
        return self.data_filled


    def split_forecasts(self):
        """
        Main function that calls all others to get answer
        
        Returns
        -------
        pd.DataFrame
            Data with shared forecast
        """
        self.check_granulatiry()
        if not self.FINAL_GRANULARITY_DELIVERED:
            self.change_granularity()
            self.share_forecast()
        
        return self.data_splitted
    
    