# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:04:35 2026

@author: Mani Subramaniyan
Utility functions related to the power gain analysis project
Update equations are based on Shi, Li & Liu 2023, SMMR 32(3):493-508

"""
import numpy as np
from statistics import NormalDist
import scipy.stats as stats

#%% Functions
def get_posterior_variance(n_0, n_h, var_0, var_h, lamb_da):
    """ 
    Update variance of current data set by borrowing information from 
    external (historical) dataset
    Inputs:
        n_0 - number of samples of current dataset
        n_h - number of samples of external or historic dataset
        var_0 - variance of current dataset
        var_h - variance of historical dataset
        lamb_da - lambda parameter for computing power prior
    Outputs:
        var_n - posterior variance (after incorporating info from external/historical dataset)
    
    """    
    var_n = 1/((n_0/var_0) + (lamb_da * n_h/var_h))
    
    return var_n
    
def get_posterior_mean(n_0, n_h, var_0, var_h, lamb_da, xbar_0, xbar_h):
    """ 
    Update mean of current data set by borrowing information from 
    external (historical) dataset
    Inputs:
        n_0 - number of samples of current dataset
        n_h - number of samples of external or historic dataset
        var_0 - variance of current dataset
        var_h - variance of historical dataset
        lamb_da - lambda parameter for computing power prior
        xbar_0 - mean of current data
        xbar_h - mean of historical dataset
    Outputs:
        mu_n - posterior mean (after incorporating info from external/historical dataset)
    
    """    
    var_n = get_posterior_variance(n_0, n_h, var_0, var_h, lamb_da)
    A = (n_0/var_0) * xbar_0 
    B = (lamb_da * n_h/var_h) * xbar_h
    mu_n = (A + B) * var_n
    
    return mu_n  
    
def get_overlap(xbar_0, xbar_h, var_0, var_h):
    """
    Compute overlap in the normal probability distributions defined by the two
    given means and variances
    
    Inputs:
        xbar_0 - mean of current data             
        xbar_h - mean of historical dataset
        var_0 - variance of current dataset 
        var_h - variance of historical dataset
    Output:
        overlap - area of overlap
    """
    A = NormalDist(xbar_0, np.sqrt(var_0))
    B = NormalDist(xbar_h, np.sqrt(var_h))
    
    return A.overlap(B)

def get_pvalue(xbar_0, xbar_h, var_0, var_h, n_0, n_h, use_sd_scale=True):
    """
    Compute the p-value for the two-sided null hypothesis that mu_0 = mu_h
    Inputs:
        xbar_0 - mean of current data        
        xbar_h - mean of historical dataset
        var_h - variance of historical dataset
        var_0 - variance of current dataset
        n_0 - number of samples of current dataset
        n_h - number of samples of external or historic dataset
        use_sd_scale - use standard deviation (not standard error) in the 
                    denomintor or scaler used for the mean difference in the
                    z-score conversion when calculating p-value
    Output:
        pvalue
    """
    
    m = np.abs(xbar_0 - xbar_h)
    if use_sd_scale: # do not scale the std with sample size
        v_0 = var_0
        v_h = var_h
    else:
        v_0 = var_0/n_0
        v_h = var_h/n_h
    std = np.sqrt(v_0 + v_h)
    z = m/std
    # 2 * (1 - cdf(z)) loses precision for large z, where 1 − cdf rounds to 0.
    # 2 * NormalDist().cdf(-z) is accurate there. It makes no practical difference
    pvalue = 2 * NormalDist().cdf(-z) # same as 2 * (1 - NormalDist().cdf(z))
    
    return pvalue

def get_sampled_variance(n, pop_var):
    """ Sample variances using Chi-square distribution
    Inputs:
        n - sample size
        pop_var - population variance
    Output:
        sv - sampled variance
    """
    df = n - 1
    chi_sample = stats.chi2.rvs(df=df)
    sv = (chi_sample * pop_var)/df
    
    return sv
