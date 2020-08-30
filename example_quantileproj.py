# -*- coding: utf-8 -*-
"""
Example file for the quantile local projection module
Romain Lafarguette, IMF, https://github.com/romainlafarguette
Time-stamp: "2020-08-30 01:50:37 Romain"
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
dates_l = [f'{y:.0f}-{q:.0f}' for y,q in zip(df['year'], df['quarter'])]
df = df.set_index(pd.to_datetime(dates_l) + pd.offsets.QuarterEnd())

# Clean some variables
df['rgdp_growth'] = df['realgdp'].rolling(4).sum().pct_change(4)
df = df.rename(columns={'infl':'inflation', 'unemp':'unemployment'})

###############################################################################
#%% Quantiles fit
###############################################################################
import quantileproj; importlib.reload(quantileproj)     # Quantile projections
from quantileproj import QuantileProj

qr = QuantileProj('inflation', ['rgdp_growth', 'unemployment'], df, horizon_l)

qr_fit = qr.fit(quantile_l=quantile_l, alpha=0.05)


qr_fit.coeffs






