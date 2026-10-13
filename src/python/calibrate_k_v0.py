# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:00:58 2026

@author: Mani Subramaniyan
Calibrate the K value in the calculation of lambda, the exponent of power prior
described in Shi, Li & Liu 2023, SMMR 32(3):493-508

"""

import numpy as np
import utils as ut
#%% Functions

def get_conf_int(k, mu_0_true, gap, var_0, var_h, n_0, n_h):
    """
    Get 95% CI for the posterior mean when we simulate different scenarios of
    separation between current and historical/external data
    Inputs:
        k - candidate calibration exponent in lambda = (A * P)^ k
        mu_0_true - the true "current data" mean used to generate the simulated data;
                    it equals mu_null (the benchmark mean) for Type I error analysis
                    and equals mu_null + delta for Power analysis. Remember, this
                    true mean is used for simulation purposes only, so its absolute
                    value can be anything, including an actually measured mean value.
                    What matters is its offset from mu_null, which the caller sets
                    and the function never sees.
        gap - gap is in current data's SD units, so the true external mean is 
                a shifted version of mu_0_true (mu_0_true + gap × sqrt(var_0)).
        var_0 - variance of current dataset
        var_h - variance of historical dataset
        n_0 - number of samples of current dataset
        n_h - number of samples of external or historic dataset  
    Output:
        ci - list of size 2; confidence interval
    
    """
    
    """
    Assuming mu_0_true as the true population mean and var_0 as the true population
    variance, sample two things - a sample mean and a sample variance - for
    both current data and history data
    """
    # Sample means
    se_0 = np.sqrt(var_0/n_0)
    se_h = np.sqrt(var_h/n_h)
    # r suffix - for emphasizing that this is randomly sampled mean
    xbar_0_r = np.random.normal(mu_0_true, se_0)
    mu_h_true = mu_0_true + gap * np.sqrt(var_0)
    xbar_h_r = np.random.normal(mu_h_true, se_h)
    # Sample variances
    var_0_r = ut.get_sampled_variance(n_0, var_0)
    var_h_r = ut.get_sampled_variance(n_h, var_h)
    # Compute overlap of the two normal distributions created by the sampled
    # means and variances
    A = ut.get_overlap(xbar_0_r, xbar_h_r, var_0_r, var_h_r) # overlap
    # Get p-value for a two-sided z-test for a null hypothesis that 
    # mu_0_true = mu_h_true
    P = ut.get_pvalue(xbar_0_r, xbar_h_r, var_0_r, var_h_r, n_0, n_h)
    # Compute lambda
    lambda_r = (A * P)**k
    # Compute posterior mean and variance
    xbar_n_r = ut.get_posterior_mean(n_0, n_h, var_0_r, var_h_r, 
                                    lambda_r, xbar_0_r, xbar_h_r)
    var_n_r = ut.get_posterior_variance(n_0, n_h, var_0_r, var_h_r, lambda_r)
    
    # Confidence interval
    ci = (np.sqrt(var_n_r) * np.array([-1, 1]) * 1.96) + xbar_n_r

    return ci    
    
    

    
    
    
    
    
    