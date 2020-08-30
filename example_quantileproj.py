# -*- coding: utf-8 -*-
"""
Example file for the quantile local projection module
Romain Lafarguette, IMF, https://github.com/romainlafarguette
Time-stamp: "2020-08-30 17:31:18 Romain"
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


ylabel = 'percentage points'
title = f'Fan chart of {self.depvar} for different horizons'
legend_loc='best'
    
# Graphic on the conditional quantiles
dsample = self.sample()

# Compute the statistics of interest
tau_l = [0.05, 0.25, 0.5, 0.75, 0.95]

dss = dsample.groupby(['horizon'])[self.depvar].quantile(tau_l)

dss = dss.reset_index().copy()
dss.columns = ['horizon', 'tau', self.depvar]
dss = dss.set_index(['horizon'])



fig, ax = plt.subplots()

ax.plot(dss.loc[dss.tau==0.05, self.depvar],
        label='5%', lw=3, color='red', ls=':')
ax.plot(dss.loc[dss.tau==0.25, self.depvar],
        label='25%', lw=2, color='black', ls='--')
ax.plot(dss.loc[dss.tau==0.50, self.depvar],
        label='Median', lw=2, color='black')
ax.plot(dss.loc[dss.tau==0.75, self.depvar],
        label='75%', lw=2, color='black', ls='--')
ax.plot(dss.loc[dss.tau==0.95, self.depvar],
        label='95%', lw=3, color='red', ls=':')

ax.fill_between(dss.loc[dss.tau==0.05, self.depvar].index,
                dss.loc[dss.tau==0.05, self.depvar],
                dss.loc[dss.tau==0.25, self.depvar],
                alpha=0.35, color='red')

ax.fill_between(dss.loc[dss.tau==0.25, self.depvar].index,
                dss.loc[dss.tau==0.25, self.depvar],
                dss.loc[dss.tau==0.75, self.depvar],
                alpha=0.75, color='red')

ax.fill_between(dss.loc[dss.tau==0.75, self.depvar].index,
                dss.loc[dss.tau==0.75, self.depvar],
                dss.loc[dss.tau==0.95, self.depvar],
                alpha=0.35, color='red')

ax.legend(framealpha=0, loc=legend_loc)
ax.set_xlabel('Horizon', labelpad=20)
ax.set_ylabel(ylabel, labelpad=20)
ax.set_title(title, y=1.02)

return(fig)


# Sampled fan chart













    




