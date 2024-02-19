#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import preprocessing
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings("ignore")


# ----------------------------------- INPUT DATA -----------------------------------  ##

data_path = r"*.csv"
df_field_merged = pd.read_csv(data_path)

model_results_path = r"*.csv"
df_function_dict = pd.read_csv(model_results_path)


# ----- SCALE DATA ----- ##

data_int = df_field_merged.copy()

numeric_cols = data_int.select_dtypes(include=[np.number]).columns
data_int_sc = data_int.copy()
data_int_sc[numeric_cols] = data_int_sc[numeric_cols].apply(stats.zscore)


# ----- MODEL RESULTS ----- ##

data_coef = []
data_y = []
data_pval = []
data_std_err = []
vif_list = []
r2_adj_list = []
residuals_list = []

X = data_int_sc[['Plant_rich', 'NDVI_avg', 'RWC', 'Clay_silt', 'pH_pH', 'Total_P_log', 'TON', 'BS_Tst', 'elevation', 'slope_log',
     'cheight', 'drought_freq', 'precip_drop', 'precip_std', 'temp_std', 'ARIDITY',
     'Plant_rich_drought_freq', 'NDVI_avg_drought_freq', 'Total_P_drought_freq', 'TON_drought_freq',
     'Clay_silt_drought_freq', 'BS_Tst_drought_freq', 'pH_drought_freq', 'precip_std_drought_freq',
     'TON_rich', 'TON_P']]


y_ = 'Dist_ratio'

var_list = []

df_function_take = df_function_dict[df_function_dict['bic'] < (df_function_dict.iloc[0]['bic'] + 4)]

for d in range(len(df_function_take)):
    s = df_function_take['predictor'].iloc[d]
    for v in range(len(s[1:-1].split(","))):
        var_list.append(s[1:-1].split(",")[v].replace(" ", "")[1:-1])


values, counts = np.unique(var_list, return_counts=True)
min_max_scaler = preprocessing.MinMaxScaler()
counts_scaled = min_max_scaler.fit_transform((counts.reshape(-1, 1)))

df_vals = pd.DataFrame([values, counts])  # , columns=[["variable","count"]])
df_vals = df_vals.T
df_vals = df_vals.rename(columns={0: 'variable', 1: 'count'}).sort_values(by=['count'], ascending=False)
df_vals['variable_importance'] = df_vals['count'] / len(df_function_take)


for f in range(len(df_function_take)):

    Y = data_int_sc[y_]
    model_var_sel = df_function_take.iloc[f]['predictor']
    model_var_sel = model_var_sel[1:-1].split(",")
    new_m = []
    for m in model_var_sel:
        new_m.append(m.replace(" ", "")[1:-1])
    selected_X = X[(new_m)]

    formula = '{} ~ '.format(y_) + ' + '.join(['%s' % variable for variable in list(new_m)])

    model = smf.glm(formula=formula, data=data_int_sc, family=sm.families.Gaussian()).fit()  ## , var_weights=np.asarray(df_field_OI.T["weight"]

    y_preds = model.predict(sm.add_constant(selected_X.values), transform=False)
    r2 = np.corrcoef(Y, y_preds)[0, 1] ** 2
    r2_adj = 1 - (1 - r2) * (len(Y) - 1) / (len(Y) - selected_X.shape[1] - 1)

    r2_adj_list.append(r2_adj)

    residuals = Y - y_preds
    residuals_list.append(residuals)

    vif_ = pd.DataFrame()
    vif_["features"] = selected_X.columns
    vif_["VIF"] = [variance_inflation_factor(selected_X.values, i) for i in range(len(selected_X.columns))]

    arr = [y_ for i in range(len(selected_X.columns) + 1)]

    data_y.append(dict(zip(selected_X.columns.insert(0, 'Intercept'), arr)))
    data_coef.append(dict(zip(selected_X.columns.insert(0, 'Intercept'), model.params)))
    data_pval.append(dict(zip(selected_X.columns.insert(0, 'Intercept'), model.pvalues)))
    data_std_err.append(dict(zip(selected_X.columns.insert(0, 'Intercept'), model.bse)))
    vif_list.append(dict(zip(selected_X.columns, vif_["VIF"])))

df_y = pd.DataFrame(data_y)
df_coef = pd.DataFrame(data_coef)
df_pval = pd.DataFrame(data_pval)
df_std_err = pd.DataFrame(data_std_err)
df_vif_list = pd.DataFrame(vif_list)
df_residuals = pd.DataFrame(residuals_list)

frames = [df_coef.mean(), df_pval.mean(), df_std_err.mean(), df_vif_list.mean()]

result_int = pd.concat(frames, axis=1).reset_index()
result_int = result_int.rename(columns={'index': "variable", 0: 'coef', 1: 'pval', 2: "std_err", 3: "VIF"})

result = pd.merge(result_int, df_vals[["variable", "variable_importance"]], on='variable', how='outer')
result = result.sort_values(by=['variable_importance'], ascending=False)


# EXPORT RESULTS ##

results_out_path = r"*.csv"
result.to_csv(results_out_path)

residuals_out_path = r"*.csv"
df_residuals.to_csv(residuals_out_path)