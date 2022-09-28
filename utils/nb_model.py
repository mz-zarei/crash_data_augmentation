#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Mohammad Zarei
# Created Date: 12 Dec 2019
# ---------------------------------------------------------------------------
"""Estimate Dispersion parameter and Fit Negative Binomial model"""
# ---------------------------------------------------------------------------
# Imports

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def computeDisperssion(data, features, y_name='Obs'):
    '''
    Returns disperssion parameter and corresponding Standard Error for a given crash "data", "features", 
    and target variable (y_name) using auxiliary Ordinary Least Square (OLS) regression without constant.

    y_name represents the target variable
    '''

    result = data.copy()
    result.reset_index(inplace=True, drop=True)
    X = sm.add_constant(data[features])
    y =  data[y_name]

    # fit a Poison model
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
    
    result['lambda'] = poisson_model.mu
    result['AUX_OLS_DEP'] = result.apply(lambda x: ((x[y_name] - x['lambda'])**2 - x['lambda']) / x['lambda'], axis=1)
    aux_olsr_results = sm.OLS(result['AUX_OLS_DEP'], result['lambda']).fit()
    alpha = aux_olsr_results.params
    alpha_se = aux_olsr_results.bse

    return alpha[0], alpha_se[0]

def fitNB(data, features, y_name='Obs'):
    '''
    Fits a NB model to the given crash "data", "features", and target variable (y_name).
    Returns fitted NB model
    '''
    
    result = data.copy()
    result.reset_index(inplace=True, drop=True)
    X = sm.add_constant(data[features])
    y =  data[y_name]

    # Estimate the dispersion parameter
    disperssion, _ = computeDisperssion(data, features, y_name)

    # Fit the NB model
    NB_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha = disperssion)).fit()

    return NB_model, disperssion
