# -*- coding: utf-8 -*-
"""
Testing file for the quantile projection module
Romain Lafarguette, rlafarguette@imf.org
Time-stamp: "2020-09-05 22:04:21 Romain"
"""
###############################################################################
#%% Preamble
###############################################################################
# Main packages
import os, sys, importlib                               # System tools
import pandas as pd                                     # Dataframes
import numpy as np                                      # Numeric tools
import statsmodels.api as sm                            # Statistics

# Graphics
import matplotlib
matplotlib.use('TkAgg') # Must be called before importing plt
import matplotlib.pyplot as plt                         # Graphical package  
import seaborn as sns                                   # Graphical tools

# Local modules
import quantileproj; importlib.reload(quantileproj)     # Quantile projections
from quantileproj import QuantileProj

###############################################################################
#%% Parameters
###############################################################################
quantile_l = list(np.arange(0.05, 1, 0.05)) # Every 5% quantiles 
horizon_l = [1, 2, 4, 8, 12] # 1Q - 3Y for quarterly data

###############################################################################
#%% Mock data
###############################################################################
df = sm.datasets.macrodata.load_pandas().data.copy()

# Create a date index with end of quarter convention
dates_l = [f'{y:.0f}-Q{q:.0f}' for y,q in zip(df['year'], df['quarter'])]
df = df.set_index(pd.to_datetime(dates_l) + pd.offsets.QuarterEnd())

# Clean some variables
df['rgdp_growth'] = df['realgdp'].rolling(4).sum().pct_change(4)
df = df.rename(columns={'infl':'inflation', 'unemp':'unemployment'})
df['current_inflation'] = df['inflation'].copy() # Avoid confusion dep/reg
print(df.describe())
print(df.tail())

###############################################################################
#%% Quantile fit
###############################################################################
import quantileproj; importlib.reload(quantileproj)     # Quantile projections
from quantileproj import QuantileProj

df['inflation_fwd1'] = df['inflation'].shift(-1)

dependent = 'inflation_fwd1'

regressors_l = ['inflation', 'rgdp_growth', 'unemployment', 'realint']

qr = QuantileProj(dependent, regressors_l, df)

qr_fit = qr.fit(quantile_l=quantile_l, alpha=0.05)

# Projection
cond_frame = df.loc[[max(df.index)], regressors_l].copy()

# Expand the conditioning frame
sim_frame_l = list()
for infl_shock in np.arange(0, 3.1, 0.1):
    sim_frame = cond_frame.copy()
    sim_frame['realint'] = cond_frame['realint'].copy() + infl_shock
    sim_frame_l.append(sim_frame)


dsim = pd.concat(sim_frame_l)

# Sample from the conditioning frame
multi_sample = qr_fit.proj(dsim).sample(seed=18021202) 

idx_d = {k:round(v,2) for k,v in enumerate(np.arange(0, 3.1, 0.1))}
multi_sample['realint_shock'] = multi_sample['conditioning'].map(idx_d)

#%%
# Compute the quantiles for each shock
ql = [0.05, 0.25, 0.5, 0.75, 0.95]

dqm = multi_sample.groupby(['realint_shock'],
                           as_index=False)['inflation_fwd1'].quantile(ql)
dqmf = dqm.reset_index().copy()
dqmf.columns=['conditioning', 'tau', 'realint_shock', 'condquant']





#%%

fig, ax = plt.subplots()
for tau in sorted(set(dqmf['tau'])):
    dd = dqmf.loc[dqmf['tau']==tau, :].copy()
    ax.plot(dd['realint_shock'], dd['condquant'], label=tau)

ax.legend()
ax.set_xlabel('Real interest shock')
ax.set_ylabel('Inflation')
ax.set_title('Linear Impact of a Real Interest Rate Shock on Inflation \n'
             'At different quantiles')    
plt.show()
plt.close('all')
















