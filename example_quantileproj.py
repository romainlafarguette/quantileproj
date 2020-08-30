# -*- coding: utf-8 -*-
"""
Example file for the quantile local projection module
Romain Lafarguette, IMF, https://github.com/romainlafarguette
Time-stamp: "2020-08-30 18:37:02 Romain"
"""

###############################################################################
#%% Modules import
###############################################################################
# Globals
import os, sys, importlib                               # System tools
import pandas as pd                                     # Dataframes
import numpy as np                                      # Numeric tools
import statsmodels.api as sm                            # Statistics

# Global plotting modules
import matplotlib.pyplot as plt                         
import seaborn as sns                                   

# Local modules
sys.path.append(os.path.abspath('modules'))             # Executable path
import quantileproj; importlib.reload(quantileproj)     # Quantile projections
from quantileproj import QuantileProj

###############################################################################
#%% Parameters
###############################################################################
quantile_l = list(np.arange(0.05, 1, 0.05)) # All 5% quantiles
horizon_l = [1, 2, 4, 8, 12] # 1Q - 2Q, 1Y, 2Y, 3Y for quarterly data

###############################################################################
#%% Example data set (statsmodels)
###############################################################################
df = sm.datasets.macrodata.load_pandas().data.copy()

# Create a date index with end of quarter convention
dates_l = [f'{y:.0f}-Q{q:.0f}' for y,q in zip(df['year'], df['quarter'])]
df = df.set_index(pd.to_datetime(dates_l) + pd.offsets.QuarterEnd())

# Clean some variables
df['rgdp_growth'] = df['realgdp'].rolling(4).sum().pct_change(4)
df = df.rename(columns={'infl':'inflation', 'unemp':'unemployment'})

###############################################################################
#%% Quantiles fit
###############################################################################
import quantileproj; importlib.reload(quantileproj)     # Quantile projections
from quantileproj import QuantileProj


dependent = 'inflation'
regressors_l = ['rgdp_growth', 'unemployment', 'realint']

qr = QuantileProj(dependent, regressors_l, df, horizon_l)

qr_fit = qr.fit(quantile_l=quantile_l, alpha=0.05)

# Design a conditioning vector (here last observation for instance)
cond_frame = df.loc[[max(df.index)], regressors_l].copy()

qr_proj = qr_fit.proj(cond_frame)



#%% Generate the plots
self = qr_proj


def cond_quantile(quantile=0.5, title=None,
                  ylabel='', legendfont=None, legendloc='best'):

    # Prepare the frame
    assert quantile in self.quantile_l, 'quantile not in quantile list'
    dcq = self.cond_quant.loc[self.cond_quant.tau==quantile, :].copy()

    # Plot
    fig, ax = plt.subplots()

    # Line
    ax.plot(dcq['horizon'], dcq['conditional_quantile_mean'],
            label=f'Conditional {100*quantile:.0f} quantile', lw=4, color='navy')
    ax.plot(dcq['horizon'], dcq['conditional_quantile_mean_ci_lower'],
            ls='--', label='Lower confidence interval', color='navy')
    ax.plot(dcq['horizon'], dcq['conditional_quantile_mean_ci_upper'],
            ls='--', label='Upper confidence interval', color='navy')

    # Area
    ax.fill_between(dcq['horizon'],
                    dcq['conditional_quantile_mean_ci_lower'],
                    dcq['conditional_quantile_mean_ci_upper'], 
                    alpha=0.15, color='dodgerblue')

    # Layout
    ax.legend(framealpha=0, loc=legend_loc, fontsize=legendfont)
    ax.set_xlabel('Horizon', labelpad=20)
    ax.set_ylabel(ylabel, labelpad=20)

    title = title or (f'Conditional {100*quantile:.0f}th quantile '
                      'over forecasting horizon')
    ax.set_title(title, y=1.02)

    return(fig)



#%%

