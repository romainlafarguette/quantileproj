# -*- coding: utf-8 -*-
"""
Quantiles Local Projections Wrapper
rlafarguette@imf.org
Time-stamp: "2020-10-15 20:23:54 Romain"
"""

###############################################################################
#%% Import
###############################################################################
# Base
import os, sys, importlib                               # System packages
import pandas as pd                                     # Dataframes
import numpy as np                                      # Numeric tools

import statsmodels as sm                                # Statistical models
import statsmodels.formula.api as smf                   # Formulas

from collections import namedtuple                      # High perf container

# Plotting packages
import matplotlib.pyplot as plt
import seaborn as sns

# Local packages
from cqsampling import inv_transform

# Warnings management
# With many quantile regressions, the convergence warnings are overwhelming
from  warnings import simplefilter                       # Filter warnings

from statsmodels.tools.sm_exceptions import (ConvergenceWarning,
                                             IterationLimitWarning)
simplefilter("ignore", category=ConvergenceWarning)
simplefilter("ignore", category=IterationLimitWarning)

# Specific warnings in quantile regressions 
np.seterr(divide='ignore', invalid='ignore')

###############################################################################
#%% Ancillary functions
###############################################################################
def zscore(series):
    """ Zscore a pandas series """
    return((series - series.mean())/series.std(ddof=0))

###############################################################################
#%% Parent class for the quantile projections
###############################################################################
class QuantileProj(object):
    """ 
    Specify a conditional quantile regression model

    Inputs
    ------
    depvar: string, 
       dependent variable 

    indvar_l: list
       list of independent variables. Intercept in included by default

    data: pd.DataFrame
       data to train the model on    

    """
    __description = "Quantile regressions wrapper"
    __author = "Romain Lafarguette, IMF, https://github.com/romainlafarguette"

    # Initializer
    def __init__(self, depvar, indvar_l, data,
                 horizon_l=[0], ):

        # Unit tests (defined at the bottom of the class)
        self.__quantilemod_unittest(depvar, indvar_l, data, horizon_l)
    
        # Attributes
        self.depvar = depvar # Dependent variable
        self.indvar_l = indvar_l
        self.horizon_l = sorted(horizon_l)

        # Data cleaning for the regression (no missing in dep and regressors)
        self.data = data[[self.depvar] + self.indvar_l].dropna().copy()
        
        # Print a warning in case of missing observations
        mn = data.shape[0] - self.data.shape[0]
        if mn > 0 : print(f'{mn:.0f} missing obs on depvar and indvar')
        
        # Create the forward variables based on the list of horizons
        self.depvar_l = list()
        for h in horizon_l:
            if h == 0:
                self.depvar_l.append(self.depvar)
            if h > 0:
                fname = f'{depvar}_fwd_{h}'
                self.depvar_l.append(fname)
                self.data[fname] = self.data[self.depvar].shift(-h)
                
        # Formula regressions for each dependent variable
        self.regform_d = {dv: self.__reg_formula(dv) for dv in self.depvar_l}
        
        # Run in parallel a zscore version of the data
        self.zdata = self.data.apply(zscore, axis=1).copy()

        
    # Class-methods (methods which returns a class defined below)    
    def fit(self, quantile_l=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            alpha=0.1):
        """ Fit the quantile regressions for each quantile, horizon """
        return(QuantileFit(self, quantile_l, alpha))
                    
    # Methods
    def __reg_formula(self, ldepvar):
        """ Generate the specification for the quantile regressions """
        # NB: I prefer formulas as the sm output is clearer
        regressors_l = self.indvar_l[0]
        for v in self.indvar_l[1:]: regressors_l += f' + {v}'
        reg_f = f'{ldepvar} ~ {regressors_l}'
        return(reg_f)
    
    def __quantilemod_unittest(self, depvar, indvar_l, data, horizon_l):
        """ Unit testing on the inputs """
        # Test for types
        assert isinstance(depvar, str), 'depvar should be string'
        assert isinstance(indvar_l, list), 'indvars should be in list'
        assert isinstance(data, pd.DataFrame), 'data should be pandas frame'
        assert isinstance(horizon_l, list), 'horizons should be in list'

        # Types and boundaries
        for var in indvar_l:
            assert isinstance(var, str), 'each indvar should be string'
                    
        for horizon in horizon_l:
            assert isinstance(horizon, int), 'horizons should be integer'
            
        # Test for consistency
        mv_l = [x for x in [depvar] + indvar_l if x not in data.columns]
        assert len(mv_l)==0, f'{mv_l} are not in data columns'


###############################################################################
#%% Class for the quantile fit
###############################################################################
class QuantileFit(object): # Fit class for the QuantileProj class

    """ 
    Fit a the quantile regressions

    Inputs
    ------
    quantile_l: list, default [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
       List of quantiles to fit the regressions upon
    
    alpha: float, default 0.05
       Level of confidence for the asymptotic test
              
    """

    # Import from QuantileProj class
    def __init__(self, QuantileProj, quantile_l, alpha): 
        self.__dict__.update(QuantileProj.__dict__) # Pass all attributes 
        self.__quantilefit_unittest(quantile_l, alpha) # Unit tests input

        # Attributes
        # Avoid floating point issue
        self.quantile_l = sorted([round(x,3) for x in quantile_l])
        self.alpha = alpha

        self.qfit_l = self.__qfit_l() # Return the fit of the qreg
        self.coeffs = self.__coeffs() # Very important: all the coefficients


        # Class-attributes (attributes from a class defined below)
        """ Plot the output of the fit """
        self.plot = QuantileFitPlot(self)
        
    def __qfit_l(self): 
        """ Fit a quantile regression at every quantile and horizon """

        # Prepare a container for each individual fit (convenient later)
        QFit =  namedtuple('Qfit', ['depvar', 'horizon', 'tau', 'qfit'])
        
        qfit_l = list() # Container
        
        for h, depvar in zip(self.horizon_l, self.depvar_l): 
            reg_f = self.regform_d[depvar] # Formula
                
            for tau in self.quantile_l: # For every tau

                # Estimate the quantile regression
                p = {'q':tau, 'maxiter':1000, 'p_tol':1e-05}
                qfit = smf.quantreg(formula=reg_f, data=self.data).fit(**p)

                # Package it into a container
                nt = {'depvar':depvar, 'horizon':h, 'tau':tau, 'qfit':qfit}
                qfit_l.append(QFit(**nt))
                
        print(f'{len(qfit_l)} quantile regressions estimated '
              f'for {len(self.horizon_l)} horizons '
              f'and {len(self.quantile_l)} quantiles')
        
        return(qfit_l)


    # Class-methods (methods which returns a class defined below)    
    def proj(self, cond_vector):
        """ Project quantiles based on a conditioning vector """
        return(QuantileProjection(self, cond_vector))

    
    def __coeffs(self):
        """ Create the frame of coefficients from all the quantile fit """

        depvar_frames_l = list() # Container        
        for qf in self.qfit_l:
            qfit = qf.qfit
            stats = [qfit.params, qfit.tvalues, qfit.pvalues,
                     qfit.conf_int(alpha=self.alpha)]
            
            stats_names = ['coeff', 'tval', 'pval', 'lower_ci', 'upper_ci']

            # Package it into a small dataframe
            dp = pd.concat(stats, axis=1); dp.columns = stats_names

            # Add information
            dp['pseudo_r2'] = qfit.prsquared
            dp.insert(0, 'tau', qf.tau)
            dp.insert(1, 'horizon', qf.horizon)

            # Store it
            depvar_frames_l.append(dp)

        # Concatenate all the frames to have a summary coefficients frame
        coeffs = pd.concat(depvar_frames_l)
        return(coeffs)


    # Unit tests
    def __quantilefit_unittest(self, quantile_l, alpha):
        """ Unit testing on the inputs """
        # Test for types
        assert isinstance(quantile_l, list), 'quantiles should be in list'

        # Test boundaries
        assert (0 < alpha < 1), 'level of confidence should be in (0,1)'
        for quantile in quantile_l:
            assert (0 < quantile < 1), 'quantile should be in (0,1)'

    

###############################################################################
#%% Projection class for the quantile fit class
###############################################################################
class QuantileProjection(object): # Projection class for the fit class

    """ 
    Project for a given conditioning vector

    Inputs
    ------
    cond_vector: Conditioning vector
                  
    """


    # Import from QuantileProj class
    def __init__(self, QuantileFit, cond_frame):
        self.__dict__.update(QuantileFit.__dict__) # Pass all attributes      
        self.__quantileproj_unittest(cond_frame) # Unit tests input
        
        # Attributes
        self.cond_frame = cond_frame.reset_index().copy() # In case of mult dim
        self.cond_quant = self.__proj_cond_quant()
        self.sample = self.sample # To have it as attribute for the plot class
        
        # Class-attributes (attributes from a class defined below)
        """ Plot the output of the projection """
        self.plot = QuantileProjectionPlot(self)
        
    # Methods    
    def sample(self, len_sample=1000, method='linear', len_bs=1000, seed=None):
        """ Sample from the conditional quantiles """
        
        dcq = self.cond_quant.copy() # Conditional quantile frame
        sample_frames_l = list() # Container
        for horizon in self.horizon_l:
            # Generate dictionary of conditional quantiles per horizon, sample
            for condition in sorted(set(dcq['conditioning'])):
                cond = ((dcq['horizon']==horizon) &
                        (dcq['conditioning']==condition))
                dcqh = dcq.loc[cond, :].copy()
                cq_d = {k:v for k,v in zip(dcqh['tau'],
                                           dcqh['conditional_quantile_mean'])}

                # Inverse transform sampling
                sample = inv_transform(cq_d, len_sample=len_sample,
                                       method=method, len_bs=len_bs, seed=seed)

                # Package it nicely
                ds = pd.DataFrame(sample, columns=[self.depvar])
                ds.insert(0, 'conditioning', condition) # Keep track
                ds.insert(1, 'horizon', horizon) # Add information
                sample_frames_l.append(ds)

        # Concatenate the frames    
        dsample = pd.concat(sample_frames_l)
        dsample = dsample.sort_values(by=['conditioning', 'horizon']).copy()
        return(dsample)

        
    def __proj_cond_quant(self):
        """ Project the conditional quantiles """
        
        dc_l = list() # Container
        for qf in self.qfit_l:
            qfit = qf.qfit
            dc = qfit.get_prediction(exog=self.cond_frame).summary_frame()
            dc.columns = ['conditional_quantile_' + x for x in dc.columns]
            dc = dc.set_index(self.cond_frame.index)

            # Add extra information
            dc.insert(0, 'conditioning', dc.index) # In case of mult dimension
            dc.insert(1, 'horizon', qf.horizon)
            dc.insert(2, 'tau', qf.tau)
            
            dc_l.append(dc) # Append to the container

        dcq = pd.concat(dc_l).reset_index(drop=True).copy()
        dcq = dcq.sort_values(by=['conditioning', 'horizon', 'tau']).copy()
        dcq = dcq.set_index(['conditioning', 'horizon', 'tau'],
                            drop=False).copy()
        return(dcq)
            
           
    # Unit tests
    def __quantileproj_unittest(self, cond_frame):
        """ Unit testing for the projection class """

        # Type testing
        c = isinstance(cond_frame, pd.DataFrame)
        assert c, 'cond_frame should be a pd.DataFrame with var in columns'

        # Test if the conditioning vector contains the independent variables
        mv_l = [x for x in self.indvar_l if x not in cond_frame.columns]
        assert len(mv_l)==0, f'{mv_l} not in conditioning frame columns'


###############################################################################
#%% Plot class for the QuantileFit class
###############################################################################
class QuantileFitPlot(object): # Plot class for the fit class

    """ 
    Plot the output of the different projections
    """

    # Import from QuantileProj class
    def __init__(self, QuantileFit):
        self.__dict__.update(QuantileFit.__dict__) # Pass all attributes


    # Methods    
    def coeffs_grid(self, horizon, title=None, num_cols=3, 
                     label_d={}, **kwds):
        """ 
        Plot the coefficients with confidence interval and R2 

        Parameters
        -----------        
        horizon: int
          Coefficients for the quantiles at a given horizon

        title: str, default 'Quantile Coefficients and Pseudo R2' 
          Sup title of the plot

        num_cols: int, default 3
          Number of columns, number of rows adjusts automatically

        label_d: dict, default empty
          Label dictionary to replace the subplots caption selectively

        """

        assert isinstance(horizon, int), 'Horizon should be integer'
        assert horizon in self.horizon_l, 'Horizon not in horizon list'
        
        # List of regressors
        var_l = ['Intercept'] + self.indvar_l
        total_plots = len(var_l) + 1 # add R2 square 

        # Compute the number of rows required
        num_rows = total_plots // num_cols

        if total_plots % num_cols >0:
            num_rows += 1 # Add one row if residuals charts
                
        # Line plot
        dc = self.coeffs.loc[self.coeffs['horizon']==horizon, :].copy()

        # Create the main figure
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True)

        axs = axs.ravel() # Very helpful !!

        # In case, customize the labels for the plots
        label_l = [None] * len(var_l)
        
        # Replace the values in list with labels_d
        for idx, var in enumerate(var_l):
            if var in label_d.keys():
                label_l[idx] = label_d[var]
            else:
                label_l[idx] = var
                               
        # Add every single subplot to the figure with a for loop
        for i, var in enumerate(var_l):
            
          # Select the data 
          dcv = dc.loc[var, :].sort_values(by='tau')
          dcv['tau'] = 100*dcv['tau'].copy() # For readibility
          
          # Main frame
          axs[i].plot(dcv.tau, dcv.coeff, lw=3, color='navy')
          axs[i].plot(dcv.tau, dcv.upper_ci, ls='--', color='blue')
          axs[i].plot(dcv.tau, dcv.lower_ci, ls='--', color='blue')

          # Fill in-between
          x = [float(x) for x in dcv.tau.values]
          u = [float(x) for x in dcv.lower_ci.values]
          l = [float(x) for x in dcv.upper_ci.values]

          axs[i].fill_between(x, u, l, facecolor='blue', alpha=0.05)

          # Hline
          axs[i].axhline(y=0, color='black', lw=0.8)

          # Caption
          axs[i].set_title(f'{label_l[i]}', y=1.02)

        # R2 plot
        dr2 = dc.loc['Intercept', :].sort_values(by='tau').copy()
        axs[len(var_l)].plot(100*dr2['tau'], dr2['pseudo_r2'].values,
                             lw=3, color='firebrick')
        axs[len(var_l)].set_title('Pseudo R2', y=1.02)
          
        # Remove extra charts
        for i in range(len(var_l) + 1, len(axs)): 
            axs[i].set_visible(False) # to remove last plot

        ttl = title or (f'Quantile coefficients at horizon {horizon} '
                        f'at {100*self.alpha:.0f}% confidence')    
        fig.suptitle(ttl)
        
        # Return both
        return(fig)


    def term_structure(self, variable, title=None, num_cols=3, 
                       label_d={}, **kwds):
        """ 
        Plot the the coefficients of a single variable across time

        Parameters
        -----------        
        variable: str
          Name of the variable to present the plot

        title: str, default 'Quantile Coefficients and Pseudo R2' 
          Sup title of the plot

        num_cols: int, default 3
          Number of columns, number of rows adjusts automatically

        label_d: dict, default empty
          Label dictionary to replace the subplots caption selectively

        """

        assert variable in self.indvar_l, 'Variable not in regressors list'
        
        # List of horizon
        total_plots = len(self.horizon_l) 

        # Compute the number of rows required
        num_rows = total_plots // num_cols

        if total_plots % num_cols >0:
            num_rows += 1 # Add one row if residuals charts
                
        # Line plot
        dc = self.coeffs.loc[variable, :].copy()

        # Create the main figure
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols)

        axs = axs.ravel() # Very helpful !!

        # In case, customize the labels for the plots
        label_l = [None] * len(self.indvar_l)
        
        # Replace the values in list with labels_d
        for idx, var in enumerate(self.indvar_l):
            if var in label_d.keys():
                label_l[idx] = label_d[var]
            else:
                label_l[idx] = var
                               
        # Add every single subplot to the figure with a for loop
        for i, horizon in enumerate(self.horizon_l):
            
          # Select the data 
          dch = dc.loc[dc['horizon']==horizon, :].sort_values(by='tau')
          dch['tau'] = 100*dch['tau'].copy() # For readibility
          
          # Main frame
          axs[i].plot(dch.tau, dch.coeff, lw=3, color='navy')
          axs[i].plot(dch.tau, dch.upper_ci, ls='--', color='blue')
          axs[i].plot(dch.tau, dch.lower_ci, ls='--', color='blue')

          # Fill in-between
          x = [float(x) for x in dch.tau.values]
          u = [float(x) for x in dch.lower_ci.values]
          l = [float(x) for x in dch.upper_ci.values]

          axs[i].fill_between(x, u, l, facecolor='blue', alpha=0.05)

          # Hline
          axs[i].axhline(y=0, color='black', lw=0.8)

          # Caption
          axs[i].set_title(f'Horizon {horizon}', y=1.02)
          
        # Remove extra charts
        for i in range(len(self.horizon_l), len(axs)): 
            axs[i].set_visible(False) # to remove last plot

        ttl = title or (f'Quantile coefficients for {variable} '
                        f'at different horizons, '
                        f'at {100*self.alpha:.0f}% confidence')    
        fig.suptitle(ttl)
        
        # Return both
        return(fig)


    def term_coefficients(self, variable,
                          tau_l=[0.05, 0.25, 0.5, 0.75, 0.95], 
                          title=None, num_cols=3, 
                          label_d={}, **kwds):
        """ 
        Plot the the coefficients of a single variable across time
        With horizon in x-axis

        Parameters
        -----------        
        variable: str
          Name of the variable to present the plot

        title: str, default 'Quantile Coefficients and Pseudo R2' 
          Sup title of the plot

        num_cols: int, default 3
          Number of columns, number of rows adjusts automatically

        label_d: dict, default empty
          Label dictionary to replace the subplots caption selectively

        """

        assert variable in self.indvar_l, 'Variable not in regressors list'

        mv_l = [x for x in tau_l if x not in self.quantile_l]
        assert len(mv_l)==0, f'{mv_l} not in quantile list'
        
        # List of horizon
        total_plots = len(tau_l) 

        # Compute the number of rows required
        num_rows = total_plots // num_cols

        if total_plots % num_cols >0:
            num_rows += 1 # Add one row if residuals charts
                
        # Line plot
        dc = self.coeffs.loc[variable, :].copy()

        # Create the main figure
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols)

        axs = axs.ravel() # Very helpful !!

        # In case, customize the labels for the plots
        label_l = [None] * len(self.indvar_l)
        
        # Replace the values in list with labels_d
        for idx, var in enumerate(self.indvar_l):
            if var in label_d.keys():
                label_l[idx] = label_d[var]
            else:
                label_l[idx] = var
                               
        # Add every single subplot to the figure with a for loop
        for i, tau in enumerate(tau_l):
            
          # Select the data 
          dch = dc.loc[dc['tau']==tau, :].sort_values(by='tau')
          
          # Main frame
          axs[i].plot(dch.horizon, dch.coeff, lw=3, color='navy')
          axs[i].plot(dch.horizon, dch.upper_ci, ls='--', color='blue')
          axs[i].plot(dch.horizon, dch.lower_ci, ls='--', color='blue')

          # Fill in-between
          x = [float(x) for x in dch.horizon.values]
          u = [float(x) for x in dch.lower_ci.values]
          l = [float(x) for x in dch.upper_ci.values]

          axs[i].fill_between(x, u, l, facecolor='blue', alpha=0.05)

          # Hline
          axs[i].axhline(y=0, color='black', lw=0.8)

          # Labels
          axs[i].set_xlabel('Horizon', labelpad=10)
          
          # Caption          
          axs[i].set_title(f'Quantile {100*tau:.0f}', y=1.02)
          
        # Remove extra charts
        for i in range(len(self.horizon_l), len(axs)): 
            axs[i].set_visible(False) # to remove last plot

        ttl = title or (f'Term quantile coefficients for {variable} '
                        f'at different quantiles, '
                        f'at {100*self.alpha:.0f}% confidence')    
        fig.suptitle(ttl)
        
        # Return both
        return(fig)

    
    
                
###############################################################################
#%% Plot class for the projection
###############################################################################
class QuantileProjectionPlot(object): # Plot class for the projection class

    """ 
    Plot the output of the different projections
    """

    # Import from QuantileProj class
    def __init__(self, QuantileProjection):
        self.__dict__.update(QuantileProjection.__dict__) # Pass all attributes

                
    # Methods
    def fitted_quantile(self, quantile=0.5, title=None,
                        ylabel='', legendfont=None, legendloc='best'):

        # Prepare the frame
        assert quantile in self.quantile_l, 'quantile not in quantile list'
        dcq = self.cond_quant.loc[self.cond_quant.tau==quantile, :].copy()

        # Plot
        fig, ax = plt.subplots()

        # Line
        ax.plot(dcq['horizon'], dcq['conditional_quantile_mean'],
                label=f'Conditional {100*quantile:.0f} quantile',
                lw=4, color='navy')
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
        ax.legend(framealpha=0, loc=legendloc, fontsize=legendfont)
        ax.set_xlabel('Horizon', labelpad=20)
        ax.set_ylabel(ylabel, labelpad=20)

        title = title or (f'Conditional {100*quantile:.0f}th quantile '
                          'over forecasting horizon')
        ax.set_title(title, y=1.02)

        return(fig)

    
    def fan_chart(self, ylabel='',
                  title=f'Fan chart at different horizons',
                  legendloc='best', legendfont=None,
                  len_sample=1000, method='linear', len_bs=1000,
                  seed=None):

        # Use the standard sample
        dsample = self.sample(len_sample=len_sample, method=method,
                              len_bs=len_bs, seed=seed)
                
        # Compute the statistics of interest
        tau_l = [0.05, 0.25, 0.5, 0.75, 0.95]

        dss = dsample.groupby(['horizon'])[self.depvar].quantile(tau_l)

        dss = dss.reset_index().copy()
        dss.columns = ['horizon', 'tau', self.depvar]
        dss = dss.set_index(['horizon'])

        # Plot
        fig, ax = plt.subplots()

        # Lines
        ax.plot(dss.loc[dss.tau==0.05, self.depvar],
                label='5%', lw=3, color='red', ls=':')
        ax.plot(dss.loc[dss.tau==0.25, self.depvar],
                label='25%', lw=2, color='black', ls='--')
        ax.plot(dss.loc[dss.tau==0.50, self.depvar],
                label='Median', lw=3, color='black')
        ax.plot(dss.loc[dss.tau==0.75, self.depvar],
                label='75%', lw=2, color='black', ls='--')
        ax.plot(dss.loc[dss.tau==0.95, self.depvar],
                label='95%', lw=3, color='red', ls=':')

        # Area
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

        # Layout
        ax.legend(framealpha=0, loc=legendloc, ncol=3, fontsize=legendfont)
        ax.set_xlabel('Horizon', labelpad=20)
        ax.set_ylabel(ylabel, labelpad=20)
        ax.set_title(title, y=1.02)

        return(fig)

        
