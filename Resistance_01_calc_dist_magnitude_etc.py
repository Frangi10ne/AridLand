#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import preprocessing
from math import log

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------- DEFINE FUNCTIONS -----------------------------------  ##


def reindex_df_year(df):
    df = (df).reset_index()
    df = df.rename(columns={"index":"year"})
    df = df.drop(['time'], axis=1)
    df['year'] = df.index + 2000
    return df


def drought_freq_func(drought_array, th_crossed_time, mult_factor):
    sum1 = drought_array[th_crossed_time-60:th_crossed_time-36].sum() * 0.01
    sum2 = drought_array[th_crossed_time-36:th_crossed_time-24].sum() * 0.25
    sum3 = drought_array[th_crossed_time-24:th_crossed_time-12].sum() * 0.5
    sum4 = drought_array[th_crossed_time-12:th_crossed_time-6].sum()  * 0.75
    sum5 = drought_array[th_crossed_time-6:th_crossed_time+1].sum()   * 1

    sum_sum = sum1 + sum2 + sum3 + sum4 + sum5

    return sum_sum*mult_factor


def Convert(lst):
    return [ -i for i in lst ]


def calculate_bic(n, mse, num_params):
    bic = n * log(mse) + num_params * log(n)
    return bic


# ----------------------------------- INPUT DATA -----------------------------------  ##

# csv file that contains NDVI data for all field plot locations with a monthly temporal resolution
df_all = pd.read_csv(r"*.csv")

# csv files that contain climatic data (precipitation, temperature and SPEI) for all field plot locations
# with a monthly temporal resolution
pts_precip_m = pd.read_csv(r'*.csv')
pts_temp_m = pd.read_csv(r'*.csv')
pts_spei6_m = pd.read_csv(r'*.csv')

# csv file that contains all measured data and additional topographic data for all plots
df_field_merged = pd.read_csv(r"*.csv")

# READ INPUT DATA #

df_all = df_all.iloc[:,1:]
df_field_merged = df_field_merged.iloc[:,1:]

df_all_columns = (df_all.columns[1:59])
plot_IDs = np.array(df_field_merged['ID'])

# calculate the yearly NDVI amplitude
df_all_yearly = df_all.copy()
df_all_yearly['time'] = pd.to_datetime(df_all.time).dt.to_period('Y').dt.to_timestamp() 
tbl_max = df_all_yearly.groupby('time')[df_all_columns].max().reset_index()
tbl_min = df_all_yearly.groupby('time')[df_all_columns].min().reset_index()
df_yearly_amplitude = reindex_df_year(tbl_max - tbl_min)


# ----------------------------------- CALCULATE DISTURBANCE MAGNITUDE ----------------------------------- ##

Dist_ratio_list  = []
NDVI_avg_list    = []

# list of the year when threshold was crossed
list_year_th = np.array(df_field_merged['crossed_yr'])

for pID, th in zip(plot_IDs, list_year_th):

    th = th-2000
    y_before = 5
    y_after = 2

    pID = str(pID)

    mean_before = (np.nanmean(df_all[pID][(th - y_before) * 12:(th) * 12]))
    mean_after = (np.nanmean(df_all[pID][th * 12:(th + y_after) * 12]))

    ampl_before = (np.nanmean(df_yearly_amplitude[pID][th - y_before:th]))
    ampl_after = (np.nanmean(df_yearly_amplitude[pID][th:th + y_after + 1]))

    NDVI_amp_mean_before = (mean_before + ampl_before) / 2
    NDVI_amp_mean_after  = (mean_after  + ampl_after)  / 2

    Dist_ratio_list.append(NDVI_amp_mean_before / NDVI_amp_mean_after)
    NDVI_avg_list.append(mean_before)

df_field_merged['Dist_ratio'] = Dist_ratio_list
df_field_merged['NDVI_avg'] = NDVI_avg_list


# ------------------------------- CALCULATE DROUGHT HISTORY AND CLIMATIC VARIABLES ------------------------------- ##

precip_std_list = []
temp_std_list = []
precip_drop = []
drought_freq_list = []

for pID in plot_IDs:

    time_th_crossed = ((df_field_merged[df_field_merged['ID'] == int(pID)]['crossed_yr'] - 2000) * 12).values[0]

    # ----- CALCULATE PRECIP AND TEMP VARIABLES ----- #

    precip_int = pts_precip_m.loc[pts_precip_m['ID'] == int(pID)]
    precip = np.array(precip_int.iloc[:, :-1]).flatten()
    temp_int = pts_temp_m.loc[pts_temp_m['ID'] == int(pID)]
    temp = np.array(temp_int.iloc[:, :-1]).flatten()

    precip_std = (np.nanstd(precip[time_th_crossed - 60:time_th_crossed + 1]))
    temp_std = (np.nanstd(temp[time_th_crossed - 60:time_th_crossed + 1]))
    precip_std_list.append(precip_std)
    temp_std_list.append(temp_std)

    pID = str(int(pID))

    mean_before = (np.nanmean(precip[time_th_crossed - 60:time_th_crossed + 1]))
    mean_after = (np.nanmean(precip[time_th_crossed:time_th_crossed + 24]))

    precip_drop.append(mean_before / mean_after)

    # ----- CALCULATE DROUGHT HISTORY ----- ##

    spei_int = pts_spei6_m.loc[pts_spei6_m['ID'] == int(pID)]
    spei = np.array(spei_int).flatten()

    drought_no = spei >= -1
    drought_moderate = ((spei < -1) & (spei >= -1.5))
    drought_severe = ((spei < -1.5) & (spei >= -2))
    drought_extreme = spei < -2

    #drought_freq = (drought_freq_func(drought_no, time_th_crossed, 0.1) +
    #                 drought_freq_func(drought_moderate, time_th_crossed,0.25) +
    #                 drought_freq_func(drought_severe, time_th_crossed, 0.5) +
    #                 drought_freq_func(drought_extreme, time_th_crossed, 1))
    #
    #drought_freq_list.append(drought_freq)

    plot_array_drought_only = spei.copy()
    plot_array_drought_only[(plot_array_drought_only > -1)] = 0
    drought_freq = drought_freq_func(plot_array_drought_only, time_th_crossed, 1)
    drought_freq_list.append(drought_freq)

df_field_merged['precip_drop'] = precip_drop
df_field_merged['precip_std']   = precip_std_list
df_field_merged['temp_std']   = temp_std_list
df_field_merged['drought_freq']   = Convert(drought_freq_list)

min_max_scaler = preprocessing.MinMaxScaler()
drought_freq_vals = np.array(df_field_merged["drought_freq"]).reshape(-1, 1)
drought_freq_vals_scaled = min_max_scaler.fit_transform(drought_freq_vals)
df_field_merged["drought_freq_sc"] =  drought_freq_vals_scaled.flatten()

# ----- TRANSFORM VARIABLES ----- ##

df_field_merged["Total_P_log"]      = np.log(df_field_merged['Total_P'])
df_field_merged["cheight_log"]      = np.log(df_field_merged['cheight'])
df_field_merged["slope_log"]        = np.log(df_field_merged['slope'])
df_field_merged["pH_pH"]            = df_field_merged['pH'] * (df_field_merged['pH'])

# ----- ADD INTERACTIONS ----- #

interaction_cols = ['Plant_rich', 'Plant_cove', 'RWC', 'NDVI_avg', 'EC', 'pH', 'Clay_silt','Total_P','TON', 'SOC', 'BS_Tst','elevation', 'slope_log', 'aspect', 'cheight', 'precip_std', 'temp_std', 'precip_drop']

for i in interaction_cols:
    df_field_merged[i + "_drought_freq"] = df_field_merged[i] * df_field_merged['drought_freq_sc']

df_field_merged["TON_rich"] = df_field_merged['TON'] * (df_field_merged['Plant_rich'])
df_field_merged["TON_P"]    = df_field_merged['TON'] * (df_field_merged['Total_P'])

save_path = r"*.csv"
df_field_merged.to_csv(save_path, index=False)
