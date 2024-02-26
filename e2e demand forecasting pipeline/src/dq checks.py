import pandas as pd
import numpy as np
import os
import sys

project_path = os.path.abspath(os.path.join('..'))

if project_path not in sys.path:
    sys.path.append(project_path)

class DQ:
    def __init__(self, check_id,
                 check_name, client,
                 input_tables, th_values,
                 lvl_data, data_path
                ):
        self.check_id = check_id
        self.check_name = check_name
        self.client = client
        self.input_tables = input_tables
        self.th_values = th_values
        self.lvl_data = lvl_data
        self.data_path = data_path
        self.data_quality_output = pd.DataFrame()
        

    def check_val_range(self, tables, th=0):
        """
        Сhecks that the column values are not greater than the specified value 
        (for example, that the prices are not negative).
        
        Parameters
        ----------
        tables : list of tuples
            List of checking tables and checking columns [(table1, col1), (table2, col2)]
        
        th : float
            Rows, where target_col less than th add to data_quality_output
            

        Returns
        -------
        pd.DataFrame
            Add rows to data_quality_output table
        """
        
        
        for table_name, target_col in tables:            
            table = pd.read_csv(self.data_path + table_name + '.csv')
            
            result = table[table[target_col] < th]
            
            if not result.empty:
                result['INPUT_COLUMN'] = target_col
                result['INPUT_TABLE'] = table_name
                result['INPUT_VALUE'] = th
                result['WARNING_TYPE'] = 'val_range'
                result['WARNING'] = f'values in column {target_col} are less than {th} in table {table_name}'
                self.data_quality_output = pd.concat([self.data_quality_output, result])

        return self.data_quality_output
    
    
    def check_cross_consistency(self, tables):
        """
        Checks that there are no key fields in the first table that are missing in the second table.
        Checking each pair of tables.
        
        Parameters
        ----------
        tables : list
            List of checking tables [table1, table2, table3]
            

        Returns
        -------
        pd.DataFrame
            Add rows to data_quality_output table
        """

        for df1_name, df2_name in list(itertools.permutations(tables, 2)):
            df1, df2 = pd.read_csv(self.data_path + df1_name + '.csv'), pd.read_csv(self.data_path + df2_name + '.csv')
            common_cols = df1.columns.intersection(df2.columns)
            common_cols = list(common_cols[common_cols.str.contains('ID')])

            if common_cols == []:
                break

            df_merged = df1.drop_duplicates(common_cols).merge(df2.drop_duplicates(common_cols), on=common_cols, 
                               how='left', indicator=True)

            result = df_merged[df_merged['_merge'] == 'left_only'][common_cols]

            if not result.empty:
                result['INPUT_TABLE'] = df1_name + ' && ' + df2_name
                result['WARNING_TYPE'] = 'cross_consistency'
                result['WARNING'] = f'id rows from table {df1_name} doesnot appear in table {df2_name}'
                self.data_quality_output = pd.concat([self.data_quality_output, result])

        return self.data_quality_output
    
    def check_time_cross_consistency(self, tables, th):
        """
        Checks tables for time cross-consistency (i.e. finding prodict_id - location_id pairs
        that have been in the SALES table and not in the STOCK table for some period)

        Parameters
        ----------
        tables : list
            List of compared tables [[table1, table2], [table3, table4]]
            
        th : int
            Threshold value showing how many rows shouldnot be in table2 to add it to data_quality_output
            

        Returns
        -------
        pd.DataFrame
            Add rows to data_quality_output table
        """
        
        for df1_name, df2_name in tables:
            df1, df2 = pd.read_csv(self.data_path + df1_name + '.csv'), pd.read_csv(self.data_path + df2_name + '.csv')
            common_cols = df1.columns.intersection(df2.columns)
            common_id_cols = common_cols[common_cols.str.contains('ID')]
            common_id_cols = list(common_id_cols[(common_id_cols.str.contains('PRODUCT')) | (common_id_cols.str.contains('LOCATION'))])

            common_dt_cols = list(common_cols[common_cols.str.endswith('_DT')])

            if common_dt_cols == [] or common_id_cols == []:
                break

            common_cols = common_id_cols + common_dt_cols

            df_merged = df1.drop_duplicates(common_cols).merge(df2.drop_duplicates(common_cols), on=common_cols, 
                               how='left', indicator=True)

            result1 = df_merged[df_merged['_merge'] == 'left_only'][common_cols]

            if not result1.empty:
                result1['INPUT_TABLE'] = df1_name + ' && ' + df2_name
                result1['INPUT_VALUE'] = th
                result1['WARNING_TYPE'] = 'time_cross_consistency'
                result1['WARNING'] = f'id rows from table {df1_name} doesnot appear in table {df2_name}'
                self.data_quality_output = pd.concat([self.data_quality_output, result1])

            both = df_merged[df_merged['_merge'] == 'both']
            both = both.groupby(common_cols).size().reset_index(name='cnt')
            result2 = both[both['cnt'] <= th].drop('cnt', axis=1)

            if not result2.empty:
                result2['INPUT_TABLE'] = df1_name + ' && ' + df2_name
                result2['INPUT_VALUE'] = th
                result2['WARNING_TYPE'] = 'time_cross_consistency'
                result2['WARNING'] = f'id rows from table {df1_name} doesnot appear in table {df2_name}'
                self.data_quality_output = pd.concat([self.data_quality_output, result2])

        return self.data_quality_output
    
    
    def format_output(self, lvl_data):
        """
        Add ID columns to data_quality_output table.

        Parameters
        ----------
        lvl_data : dict
            Information about ID lvls (LOCATION, CUSTOMER, etc.)
            

        Returns
        -------
        pd.DataFrame
            Formatted data_quality_output table
        """
        for el in lvl_data.keys():
            if f'{el}_ID' not in self.data_quality_output.columns:
                continue
                
            df = pd.read_csv(self.data_path + lvl_data[el] + '.csv')
            cols = df.columns
            last_lvl = len(cols[cols.str.contains(f'{el}_LVL_ID')])

            self.data_quality_output[f'{el}_LVL_ID{last_lvl + 1}'] = self.data_quality_output[f'{el}_ID'].astype('Int64')
            self.data_quality_output[f'{el}_LVL'] = last_lvl + 1
            self.data_quality_output = self.data_quality_output.drop(f'{el}_ID', axis=1)
            
    
    def check(self):
        self.check_val_range(
            self.input_tables['val_range'],
            self.th_values['val_range']
        )
        self.check_cross_consistency(self.input_tables['cross_consistency'])
        
        self.check_time_cross_consistency(self.input_tables['time_cross_consistency'],
                                          self.th_values['time_cross_consistency']
                                         )
        
        self.format_output(self.lvl_data)
