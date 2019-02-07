# Load and clean raw dataset from 'data/raw' folder 
    # Cleaned data stored in 'data/interim' folder 

# Imports (External)
import numpy as np
import pandas as pd
import datetime as dt
import xlrd
import xlsxwriter
from collections import OrderedDict

import sys
sys.path.append('../')  

# Load in excel file and map each excel sheet to an ordered dict
raw_xlsx_file = pd.ExcelFile("../data/raw/raw_data.xlsx")
dict_dataframes = pd.read_excel(raw_xlsx_file,sheet_name = None)
#print(type(dict_dataframes))

# Convert ordered of dataframes to regular dict
dict_dataframes = dict(dict_dataframes)
#print(type(dict_dataframes)()

# Convert all sheet names/dict keys to lowercase using list comprehension 
    # Source: https://stackoverflow.com/a/38572808
dict_dataframes = {k.lower(): v for k, v in dict_dataframes.items()}

# Print name + number of sheets in dict of dataframes:
#print("Number of sheets: ",len(dict_dataframes),"\n")
#print("\n".join(list(dict_dataframes.keys())))

# Panel A, Developing Market
    # 'csi300 index data',
    # 'csi300 index future data'
    # 'nifty 50 index data'
    # 'nifty 50 index future data'
# Panel B, Relatively Developed Market
    # 'hangseng index data'
    # 'hangseng index future data'
    # 'nikkei 225 index data'
    # 'nikkei 225 index future data'
# Panel C, Developed Market
    # 's&p500 index data'
    # 's&p500 index future data'
    # 'djia index data'
    # 'djia index future data'

# Rename all dataframe column headers in each dataframe in dict_dataframes to lowercase
for item in dict_dataframes:
    dict_dataframes[item].columns = map(str.lower, dict_dataframes[item].columns)

# Convert dict back to orderdict after reorder to match Panel A/B/C format
    # Source: https://stackoverflow.com/a/46447976
key_order = ['csi300 index data',
'csi300 index future data',
'nifty 50 index data',
'nifty 50 index future data',
'hangseng index data',
'hangseng index future data',
'nikkei 225 index data',
'nikkei 225 index future data',
's&p500 index data',
's&p500 index future data',
'djia index data',
'djia index future data',
]
list_of_tuples = [(key, dict_dataframes[key]) for key in key_order]
dict_dataframes = OrderedDict(list_of_tuples)

# Obtain information on each sheet (row and column info)
# for item in dict_dataframes:
#     # Obtain number of rows in dataframe
#     #rc=dict_dataframes[item].shape[0]
#     # Obtain number of columns in dataframe
#     #cc =  len(dict_dataframes[item].columns)
#     print ("=======================================")
#     print (item,"\n")
#     print (dict_dataframes[item].info(verbose=False))

# Drop column 'matlab_time' from all dataframes in OrderedDict + rename OHLC columns for consistency
for item in dict_dataframes:
    for subitem in dict_dataframes[item]:
        if 'matlab_time' in subitem:
            print(subitem,"Dropped from ", item)
            dict_dataframes[item].drop(subitem,axis=1, inplace=True) 
        # Rename OHLC columns for consistency
        if 'open price' in subitem:
            print(subitem,"Renamed from ", item)
            dict_dataframes[item].rename(columns={'open price':'open'},inplace=True)
        if 'high price' in subitem:
            print(subitem,"Renamed from ", item)
            dict_dataframes[item].rename(columns={'high price':'high'},inplace=True)
        if 'low price' in subitem:
            print(subitem,"Renamed from ", item)
            dict_dataframes[item].rename(columns={'low price':'low'},inplace=True)
        if 'closing price' in subitem:
            print(subitem,"Renamed from ", item)
            dict_dataframes[item].rename(columns={'closing price':'close'},inplace=True)
        if 'close price' in subitem:
            print(subitem,"Renamed from ", item)
            dict_dataframes[item].rename(columns={'close price':'close'},inplace=True)     

# Rename date/ntime columns to date + drop mislabeled matlab_time columns
dict_dataframes['csi300 index data'].rename(columns={'time':'date'},inplace=True)
dict_dataframes['csi300 index future data'].rename(columns={'num_time':'date'},inplace=True)

dict_dataframes['nifty 50 index data'].drop(columns=['ntime'],axis=1, inplace=True)
dict_dataframes['nifty 50 index future data'].drop(columns=['ntime'],axis=1, inplace=True)

dict_dataframes['hangseng index data'].drop(columns=['time'],axis=1, inplace=True)
dict_dataframes['hangseng index data'].rename(columns={'ntime':'date'},inplace=True)

dict_dataframes['hangseng index future data'].rename(columns={'ntime':'date'},inplace=True)

dict_dataframes['nikkei 225 index data'].rename(columns={'ntime':'date'},inplace=True)
dict_dataframes['nikkei 225 index data'].drop(columns=['time'],axis=1, inplace=True)

dict_dataframes['nikkei 225 index future data'].drop(columns=['time'],axis=1, inplace=True)
dict_dataframes['nikkei 225 index future data'].rename(columns={'ntime':'date'},inplace=True)

dict_dataframes['s&p500 index data'].drop(columns=['time'],axis=1, inplace=True)
dict_dataframes['s&p500 index data'].rename(columns={'ntime':'date'},inplace=True)

dict_dataframes['djia index data'].drop(columns=['time'],axis=1, inplace=True)
dict_dataframes['djia index data'].rename(columns={'ntime':'date'},inplace=True)

dict_dataframes['djia index future data'].drop(columns=['time'],axis=1, inplace=True)

# # Verify date rename + column drop/rename
# for item in dict_dataframes:
#     # Obtain number of rows in dataframe
#     rc=dict_dataframes[item].shape[0]
#     # Obtain number of columns in dataframe
#     cc =  len(dict_dataframes[item].columns)
#     print ("=======================================")
#     print (item,"\n")
#     print (dict_dataframes[item].info(verbose=False))

# Save cleaned data to disk (both index data and futures in one xlsx sheet)
def frames_to_excel(df_dict, path):
    # frames_to_excel() source: https://stackoverflow.com/q/51696940
    """Write dictionary of dataframes to separate sheets, within 
        1 file."""
    writer = pd.ExcelWriter(path, engine='xlsxwriter')
    for tab_name, dframe in df_dict.items():
        dframe.to_excel(writer, sheet_name=tab_name)
    writer.save() 
    
frames_to_excel(dict_dataframes,"../data/interim/clean_data.xlsx")

# Save clean data to disk - index data only
key_order = ['csi300 index data',
'nifty 50 index data',
'hangseng index data',
'nikkei 225 index data',
's&p500 index data',
'djia index data',
]
list_of_tuples = [(key, dict_dataframes[key]) for key in key_order]
dict_dataframes_index = OrderedDict(list_of_tuples)

frames_to_excel(dict_dataframes_index,"../data/interim/clean_data_index.xlsx")

# Save clean data to disk - future data only
key_order = [
'csi300 index future data',
'nifty 50 index future data',
'hangseng index future data',
'nikkei 225 index future data',
's&p500 index future data',
'djia index future data',
]
list_of_tuples = [(key, dict_dataframes[key]) for key in key_order]
dict_dataframes_futures = OrderedDict(list_of_tuples)

frames_to_excel(dict_dataframes_futures,"../data/interim/clean_data_futures.xlsx")