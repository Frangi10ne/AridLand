#!/usr/bin/env python
# coding: utf-8

import scipy.stats as stats
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm

import warnings
warnings.filterwarnings("ignore")

## ----------------------------------- DEFINIE FUNCTIONS -----------------------------------  ##

def fit_function_ext(x, y, degree=3, plot_p=True):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    X = x[:, np.newaxis]
    summary = model.fit(X, y)
    y_plot = model.predict(X)

    p_val = 0

    if plot_p == True:
        X2 = sm.add_constant(X)
        est = sm.OLS(y, X2)
        est2 = est.fit()

        p_val = est2.summary2().tables[1]['P>|t|']['x1']

    return y_plot, p_val


# ----------------------------------- INPUT DATA -----------------------------------  #


data_path = r"*.csv"
model_results_path = r"*.csv"
model_residuals_path = r"*.csv"

df_field_merged = pd.read_csv(data_path)
result = pd.read_csv(model_results_path)
df_residuals = pd.read_csv(model_residuals_path)
df_residuals = df_residuals.iloc[:,1:]

# ----- SCALE DATA ----- #

data_int = df_field_merged.copy()

numeric_cols = data_int.select_dtypes(include=[np.number]).columns
data_int_sc = data_int.copy()
data_int_sc[numeric_cols] = data_int_sc[numeric_cols].apply(stats.zscore)

# ----- PLOT RESULTS ----- #

var_list     = ['Plant_rich', 'NDVI_avg', 'cheight', 'BS_Tst', 'RWC',
                'precip_drop', 'precip_std', 'temp_std', 'ARIDITY', 'drought_freq', 
                'Clay_silt', 'TON', 'Total_P_log', 'pH_pH', 'slope_log', 'elevation']

var_list_leg = ['Plant richness', 'Average NDVI', 'Canopy height', 'Bare soil fraction', 'Relative woody cover',
                'Precipitation\ndifference', 'Precipitation\nvariability', 'Temperature\nvariability', 'Aridity', 'Drought frequency', 
                'Soil texture', 'Nitrogen', 'Phosphorus', 'pH value', 'Slope', 'Elevation']

save_as_img  = r"*.jpg"

fig = plt.figure(figsize=(30,18)) 
gs = gridspec.GridSpec(3, 6)
gs.update(wspace=0.03)
gs.update(hspace=0.5)
ax1 = plt.subplot(gs[0, 0:1]) 
ax2 = plt.subplot(gs[0, 1:2], sharey=ax1) 
ax3 = plt.subplot(gs[0, 2:3], sharey=ax1) 
ax4 = plt.subplot(gs[0, 3:4], sharey=ax1) 
ax5 = plt.subplot(gs[0, 4:5], sharey=ax1) 

ax6 = plt.subplot(gs[1, 0:1], sharey=ax1) 
ax7 = plt.subplot(gs[1, 1:2], sharey=ax1) 
ax8 = plt.subplot(gs[1, 2:3], sharey=ax1) 
ax9 = plt.subplot(gs[1, 3:4], sharey=ax1) 
ax10 = plt.subplot(gs[1, 4:5], sharey=ax1) 

ax11 = plt.subplot(gs[2, 0:1], sharey=ax1) 
ax12 = plt.subplot(gs[2, 1:2], sharey=ax1) 
ax13 = plt.subplot(gs[2, 2:3], sharey=ax1) 
ax14 = plt.subplot(gs[2, 3:4], sharey=ax1) 
ax15 = plt.subplot(gs[2, 4:5], sharey=ax1) 
ax16 = plt.subplot(gs[2, 5:6], sharey=ax1) 

axes_list = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16)   

for var, var_leg, a in zip(var_list, var_list_leg, axes_list):

    var_list_results = result["variable"][:-1]
    var_coef         = result["coef"]

    df_plot          = data_int_sc[var_list_results]
    df_plot['drought_freq']     = data_int['drought_freq_sc']
    df_plot['drought_freq_cat'] = np.where(df_plot['drought_freq'] <= 0.5, 'low', "no_data")
    df_plot['drought_freq_cat'] = np.where(df_plot['drought_freq'] > 0.5, 'high', df_plot['drought_freq_cat'])

    df_plot['residuals'] = df_residuals.mean().tolist()

    idx = 0
    coef = 0
    
    if not var in var_list_results.values:

        df_plot = data_int_sc[['Plant_rich', 'NDVI_avg', 'RWC', 'Clay_silt', 'pH_pH', 'Total_P_log', 'TON', 'BS_Tst','elevation', 'slope_log', 'cheight', 'drought_freq', 'precip_drop', 'precip_std', 'temp_std', 'ARIDITY',
                 'Plant_rich_drought_freq', 'NDVI_avg_drought_freq', 'Total_P_drought_freq', 'TON_drought_freq', 'Clay_silt_drought_freq', 'BS_Tst_drought_freq', 'pH_drought_freq', 'precip_std_drought_freq', 
                 'TON_rich', 'TON_P']]

        df_plot['drought_freq'] = data_int['drought_freq_sc']
        df_plot['drought_freq_cat'] = np.where(df_plot['drought_freq'] <= 0.5, 'low', "no_data")
        df_plot['drought_freq_cat'] = np.where(df_plot['drought_freq'] > 0.5, 'high', df_plot['drought_freq_cat'])

        df_plot['residuals'] = df_residuals.mean().tolist()
        
    else:
        idx = list(var_list_results.values).index(var)
        coef = var_coef.iloc[idx]


    for co, c in zip(['low', 'high'], ['#41ab5d', '#c51b7d']):
        
        df_plot_co = df_plot[df_plot['drought_freq_cat'] == co]
        x_co = np.array(df_plot_co[var])

        # PARTIAL RESIDUALS = model residuals + independent variable * model coefficient
        y_co = np.array(df_plot_co['residuals'] + (x_co*coef))

        # IF the dependent variable has an interaction term:
        if var + "_" + 'drought_freq':
            p = result[result["variable"] == var + "_" + 'drought_freq']['pval']

            # IF the independent variable has significant interactions with drought frequency:
            if p.values<0.05:
                x_co2 = np.array(df_plot_co[var + "_" + 'drought_freq'])
                coef2 = np.array(result[result["variable"] == var + "_"  + 'drought_freq']['coef'])

                # PARTIAL RESIDUALS = model residuals + independent variable * model coefficient + independent variable's interaction * model coefficient of this interaction term
                y_co = np.array(df_plot_co['residuals'] + (x_co*coef) + (x_co2*coef2))

            elif p.values>0.05:
                y_co = np.array(df_plot_co['residuals'] + (x_co*coef))

        scatter = sns.scatterplot(x=x_co, y=y_co*(-1), color=c, s=100, alpha=0.75, edgecolor='black', linewidth=1, legend = False, ax=a)

        y_fit_co, p_val = fit_function_ext(x_co, y_co, degree=1, plot_p = True)
        a.plot(x_co, y_fit_co*(-1), "-", color = c, linewidth=1.8, alpha=0.9, label= "{:.4f}".format(p_val))

        a.set_ylim(-3.2, 3.2)

        a.axhline(y=0, ls="--", lw=0.5, c="black")
        a.tick_params(axis='x', rotation=0, labelsize=16)
        a.tick_params(axis='y', rotation=0, labelsize=0)
        a.set_xlabel(r"{}".format(var_leg), fontsize=22)

        if a == ax1:
            a.set_ylabel(r"Vegetation resistance", fontsize=22)
            a.tick_params(axis='y', rotation=0, labelsize=20)

        a.legend([],[], frameon=False)


save_as_img = r"*.jpeg"
plt.savefig(save_as_img, bbox_inches='tight', format="jpg", dpi=300)

plt.show()
