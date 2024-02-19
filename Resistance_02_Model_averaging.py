#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import scipy.stats as stats
from datetime import date, datetime
import itertools
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error
from math import log

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------- DEFINIE FUNCTIONS -----------------------------------  ##


def calculate_bic(n, mse, num_params):
    bic = n * log(mse) + num_params * log(n)
    return bic


# ----------------------------------- INPUT DATA -----------------------------------  ##


data_path = r"*.csv"
df_field_merged = pd.read_csv(data_path)


# ----- SCALE DATA ----- ##

data_int = df_field_merged.copy()

numeric_cols = data_int.select_dtypes(include=[np.number]).columns
data_int_sc = data_int.copy()
data_int_sc[numeric_cols] = data_int_sc[numeric_cols].apply(stats.zscore)


# ----- RUN MODELS ----- ##

X = data_int_sc[['Plant_rich', 'NDVI_avg', 'RWC', 'Clay_silt', 'pH_pH', 'Total_P_log', 'TON', 'BS_Tst', 'elevation', 'slope_log',
     'cheight', 'drought_freq', 'precip_drop', 'precip_std', 'temp_std', 'ARIDITY',
     'Plant_rich_drought_freq', 'NDVI_avg_drought_freq', 'Total_P_drought_freq', 'TON_drought_freq',
     'Clay_silt_drought_freq', 'BS_Tst_drought_freq', 'pH_drought_freq', 'precip_std_drought_freq',
     'TON_rich', 'TON_P']]


function_dict = {'dep_var': [], 'predictor': [], 'predictor_len': [], 'r-squared': [], 'r-squared_adj': [], 'bic': []}

iter_range = np.arange(4, 11)

today = date.today()
print(today.strftime("%d/%m/%Y"))

y_var = 'Dist_ratio'

print("Start:", datetime.now().strftime("%H:%M:%S"))

Y = data_int_sc[y_var]

for i in iter_range:
    col_pairs = list(itertools.combinations(X.columns, i))
    for p in col_pairs:

        # Test to sort out the combinations that contain variable combinations that cause VIF to be higher than 4/ 5
        selected_X = X[list(p)]

        VIF = [variance_inflation_factor(selected_X.values, i) for i in range(len(selected_X.columns))]

        if all(item <= 8 for item in VIF):
            function_dict['dep_var'].append(y_var)

            formula = '{} ~ '.format(y_var) + ' + '.join(['%s' % variable for variable in list(p)])
            model = smf.glm(formula=formula, data=data_int_sc, family=sm.families.Gaussian()).fit()

            y_preds = model.predict(sm.add_constant(selected_X.values), transform=False)

            # Add the predictor variable names to our dictionary
            function_dict['predictor'].append(p)
            function_dict['predictor_len'].append(len(p))

            r2 = np.corrcoef(Y, y_preds)[0, 1] ** 2
            r2_adj = 1 - (1 - r2) * (len(Y) - 1) / (len(Y) - selected_X.shape[1] - 1)

            # Calculate BIC
            num_params = len(p)
            mse = mean_squared_error(Y, y_preds)
            bic = calculate_bic(len(Y), mse, num_params)

            # Add the r-squared value to our dictionary
            function_dict['r-squared'].append(r2)
            function_dict['r-squared_adj'].append(r2_adj)
            function_dict['bic'].append(bic)

today = date.today()
print(today.strftime("%d/%m/%Y"))
print("Done:", datetime.now().strftime("%H:%M:%S"))

df_function_dict = pd.DataFrame.from_dict(function_dict)
df_function_dict = df_function_dict.sort_values(by=['bic'], ascending=True)

save_model_results = r"*.csv"
df_function_dict.to_csv(save_model_results)
