# -*- coding: utf-8 -*-
"""
Inverse Transform Sampling from Conditional Quantiles with Uncrossing
Romain Lafarguette, https://github.com/romainlafarguette
June 2018
Time-stamp: "2020-08-30 16:30:50 Romain"
"""

###############################################################################
#%% Imports
###############################################################################
# Modules
import numpy as np                                      # Numeric tools

# Methods
from scipy import interpolate                           # Linear interpolation

# Functions
from scipy.stats import t  # Tdistribution
from scipy.stats import norm  # Gaussian distribution

###############################################################################
#%% Quantile Interpolation following Schmidt and Zhu (2016)
###############################################################################
def quantile_interpolation(alpha, cond_quant_dict):
    """ 
    Quantile interpolation function, following Schmidt and Zhu (2016) p12
    - Alpha is the quantile that needs to be interpolated
    - cond_quant_dict is the dictionary of quantiles to interpolate on 
    Return:
    - The interpolated quantile
    """

    # List of quantiles
    qlist = sorted(list(cond_quant_dict.keys()))
    min_q = min(qlist)
    max_q = max(qlist)

    # Fix the base quantile function (usually a N(0,1))
    base = norm.ppf

    # Considering multiple cases
    if alpha in qlist: ## No need to interpolate, just on the spot !!
        interp = cond_quant_dict[alpha]

    elif alpha < min_q: ## The left edge
        ## Compute the slope (page 13) 
        b1_up = (cond_quant_dict[max_q] - cond_quant_dict[min_q])
        b1_low = base(max_q) - base(min_q)
        b1 = b1_up/b1_low

        ## Compute the intercept (page 12)
        a1 = cond_quant_dict[min_q] - b1*base(min_q)

        ## Compute the interpolated value
        interp = a1 + b1*base(alpha)
        
    elif alpha > max_q: # The right edge (same formula)
        # Compute the slope (page 13) 
        b1_up = (cond_quant_dict[max_q] - cond_quant_dict[min_q])
        b1_low = base(max_q) - base(min_q)
        b1 = b1_up/b1_low

        # Compute the intercept (page 12)
        a1 = cond_quant_dict[min_q] - b1*base(min_q)

        # Compute the interpolated value
        interp = a1 + b1*base(alpha)

    else: # In the belly
        # Need to identify the closest quantiles
        local_min_list = [x for x in qlist if x < alpha]
        local_min = max(local_min_list) # The one immediately below

        local_max_list = [x for x in qlist if x > alpha]
        local_max = min(local_max_list) # The one immediately above

        # Compute the slope
        b_up = (cond_quant_dict[local_max] - cond_quant_dict[local_min])
        b_low = base(local_max) - base(local_min)
        b = b_up/b_low

        # Compute the intercept
        a = cond_quant_dict[local_max] - b*base(local_max)
        
        # Compute the interpolated value
        interp = a + b*base(alpha)

    # Return the interpolated quantile    
    return(interp)

###############################################################################
#%% Uncrossing using Cherzonukov et al (Econometrica 2010)
###############################################################################
def quantiles_uncrossing(cond_quant_dict, method='linear', len_bs=1000,
                         seed=None):
    """ 
    Uncross a set of conditional_quantiles either via Cherzonukov et al (2010)
    (bootstrapped rearrangement) or Schmidt and Zhu (functional interpolation)
    
    Input:
    - A dictionary of quantile: conditional quantiles
    - Interpolation method: either linear or probabilistic. 
    The probabilistic quantile interpolation follows Zhu and Schmidt 2016
    - len_bs: length of the bootstrapped rearrangement
    Output:
    - A dictionary of quantile: uncrossed conditional quantiles
    """
    __description = "Quantiles uncrossing, following Cherzonukov et al. (2010)"
    __author = "Romain Lafarguette, IMF, rlafarguette@imf.org"
    
    ## List of quantiles
    ql = sorted(list(cond_quant_dict.keys()))
    
    cond_quant = [cond_quant_dict[q] for q in ql] # Because dict is not ordered

    # Treatment of the seed for the random numbers generator
    if seed: # If seed fixed, then use it
        np.random.seed(seed)
    else: # Else, use default
        pass
    
    # Uncrossing
    if sorted(cond_quant) == cond_quant: # Check if the quantiles are crossed
        #print('Conditional quantiles already sorted !')
        cond_quant_uncrossed_dict = cond_quant_dict
        
    else: # Uncross them using either of the two methods
        if method=='linear': # Chernozukov et al (Econometrica, 2010)        
            inter_lin = interpolate.interp1d(ql, cond_quant,
                                             fill_value='extrapolate')

            # Bootstrap the quantile function
            bootstrap_qf = inter_lin(np.random.uniform(0, 1, len_bs))

            # Now compute the percentiles of the bootstrapped quantiles 
            cond_quant_uncrossed = [np.percentile(bootstrap_qf, 100*q)
                                    for q in ql]

            # They are the uncrossed quantiles ! (super simple)
            cond_quant_uncrossed_dict = dict(zip(ql, cond_quant_uncrossed))

        elif method=='probabilistic': # Use Schmidt and Zhu (2016) approach
            bootstrap_qf = [quantile_interpolation(u, cond_quant_dict)
                            for u in np.random.uniform(0, 1, len_bs)]

            # Now compute the percentiles of the bootstrapped quantiles 
            cond_quant_uncrossed = [np.percentile(bootstrap_qf, 100*q)
                                    for q in ql]

            # They are the uncrossed quantiles !
            cond_quant_uncrossed_dict = dict(zip(ql, cond_quant_uncrossed))

        else:
            raise ValueError('Interpolation method misspecified')
            
    # Return the uncrossed quantiles    
    return(cond_quant_uncrossed_dict)


###############################################################################
#%% Inverse transform sampling function
###############################################################################
def inv_transform(cond_quant_dict, len_sample=1000, method='linear',
                  len_bs=1000, seed=None):
    """ 
    Sample a list of conditional quantiles via inverse transform sampling

    Parameters
    ----------
      cond_quant_dict : dictionary     
        Dictionary of conditional quantiles as {quantile: conditional quantile}

      quantile_list: list
        List of quantiles (should match cond_quant)

      len_sample: integer, default=1000
        Length of the sample to be returned

      len_bs=integer, default=1000
        Length of the bootstrap for quantiles uncrossing

      seed:integer, default=None
        Seed of the random numbers generator, for replicability purposes


    Returns
    -------
      A sample of length len_sample, with the quantile matching cond_quant

    """

    __description = "Inverse Transform Sampling from Conditional Quantiles"
    __author = "Romain Lafarguette, IMF, rlafarguette@imf.org"


    # Treatment of the seed for the random numbers generator
    if seed: # If seed fixed, then use it
        np.random.seed(seed)
    else: # Else, use default
        pass
        
    # Uncross the conditional quantiles if necessary
    # Rearrangement either Cherzonukov et al. (2010) or Schmidt and Zhu (2016)
    u_cond_quant_dict = quantiles_uncrossing(cond_quant_dict,
                                             method=method,
                                             len_bs=1000, seed=None)
    
    # Extract lists from dictionary
    quantile_list = sorted(u_cond_quant_dict.keys())
    num_quantile_list = [np.float(q) for q in quantile_list]
    u_cond_quant = [u_cond_quant_dict[k] for k in quantile_list] 

    # Set up an interpolation of the empirical cdf
    inv_ecdf = interpolate.interp1d(num_quantile_list, u_cond_quant,
                                    fill_value='extrapolate')

    # Draw a random sample
    U = np.random.rand(len_sample)
    
    # Use the inv_ecdf to invert a random uniform sample
    sample = inv_ecdf(U)
    
    return(sample) 


###############################################################################
###############################################################################
#%% Miscellaneous
###############################################################################
###############################################################################


###############################################################################
#%% Random sampling of a matrix, by columns
###############################################################################
def sample_cols(X, num_reps=1000):
    """ 
    Uncross a set of conditional_quantiles using Cherzonukov et al. (2010)
    Via bootstrapped rearrangement
    
    Parameters
      X : numpy array     
        The matrix to be reshuffled
      num_reps: integer; default:1000
        number of replications (columns of a new matrix)
    Returns
      A resampled matrix, of size (X.rows, num_reps)
    """

    __description = "Random sampling of a matrix, by columns"
    __author = "Romain Lafarguette, IMF, rlafarguette@imf.org"
        
    R = X.shape[0] # Number of rows
    C = X.shape[1] # Number of columns
    
    ## Create the index for rows and columns
    rows_ind = [[x for x in range(R)] for _ in range(num_reps)]
    cols_ind = [np.random.choice(C, R) for _ in range(num_reps)]

    ## Zip the rows and columns index
    idx_list = [(x,y) for x,y in zip(rows_ind, cols_ind)]

    ## Create the resampled matrix, including the intercept
    XR = np.empty((R, num_reps))

    for col in range(num_reps):
        XR[:, col] = X[idx_list[col]]

    return(XR)    
