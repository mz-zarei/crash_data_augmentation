#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Mohammad Zarei
# Created Date: 12 Dec 2019
# ---------------------------------------------------------------------------
"""Simulate Crash Data Points From Negative Binomial Distribution"""
# ---------------------------------------------------------------------------
# Imports
import pandas as pd
import numpy as np


def simulateCrashData(coef_vector, data_size=1000, error_mean=1, error_var=1, constant=0.5):
    '''
    Returns a dataframe with simulated features, expected crash count (lambda), and observed crash count (Obs) 
    sampled from Poison-Gamma Distribution with log-linear relationship between features and crash counts:

    Obs ~ Poison(lambda)
    lambda = exp(constant + B1 * X1 + ... + Bn * Xn + e)
    exp(e) ~ Gamma(error_mean, error_var)
    X1,..,Xn ~ Uniform(0,1)
    
    coef_vector: [B1, B2, ..., Bn]

    return DataFrame  
    '''

    Obs, features, Lambda, feature_name = [], [], [], []
    coef_vector = np.array(coef_vector)
    
    for i in range(data_size):
        # sample X vector from uniform dist
        X = np.random.uniform(0,1,size=len(coef_vector))

        scale = error_var/error_mean
        shape = error_mean/scale

        error_term = np.random.gamma(shape = shape, scale=scale)     

        ro = np.exp(np.dot(coef_vector,X) + constant)
        lambda_true = ro * error_term
        Lambda.append(lambda_true)
        Obs.append(np.random.poisson(lam = lambda_true))
        features.append(X)
    
    for i in range(1,len(coef_vector)+1):
        feature_name.append('X'+str(i))
    simulated_data = pd.DataFrame(features, columns = feature_name)
    simulated_data['Obs'] = Obs
    simulated_data['Lambda']  = Lambda
    
    return simulated_data

def simulateCrashData_NL(coef_vector, data_size=1000, error_mean=1, error_var=1, constant=0.5):
    '''
    Returns a dataframe with simulated features, expected crash count (lambda), and observed crash count (Obs)
    sampled from Poison-Gamma Distribution with fixed log-nonlinear relationship between features and crash counts:

    Y ~ Poison(lambda)
    lambda = exp(constant + B1*X1^2 + B2*X2*X2 + B3*X3^0.5 + B4*(X4*X2)^0.25  + e)
    exp(e) ~ Gamma(error_mean, error_var)
    X1,..,Xn ~ Uniform(0,1)
    
    coef_vector: [B1, B2, ..., Bn] 
    '''

    Obs, features, Lambda, feature_name = [], [], [], []
    coef_vector = np.array(coef_vector)
    
    for i in range(data_size):
        # sample X vector from uniform dist
        X = np.random.uniform(0,1,size=len(coef_vector))

        scale = error_var/error_mean
        shape = error_mean/scale

        error_term = np.random.gamma(shape = shape, scale=scale)  
        X_nl = [X[0]**2, X[0]*X[1], X[2]**0.5, (X[3]*X[1])**0.25 ]

        ro = np.exp(np.dot(coef_vector,X_nl) + constant)
        lambda_true = ro * error_term
        Lambda.append(lambda_true)
        Obs.append(np.random.poisson(lam = lambda_true))
        features.append(X)
    
    for i in range(1,len(coef_vector)+1):
        feature_name.append('X'+str(i))
    simulated_data = pd.DataFrame(features, columns = feature_name)
    simulated_data['Obs'] = Obs
    simulated_data['Lambda']  = Lambda
    
    return simulated_data

def simulateCrashData_X(X_data, coef_vector, replacement = False, data_size=1000, error_mean=1, error_var=1, constant=0.5):
    '''
    Returns a dataframe with given X features (X_data), expected crash count (lambda), and observed crash count (Obs)
    sampled from Poison-Gamma Distribution with fixed log-linear relationship between features and crash counts:

    Y ~ Poison(lambda)
    lambda = exp(constant + B1 * X1 + ... + Bn * Xn + e)
    exp(e) ~ Gamma(error_mean, error_var)
    
    coef_vector: [B1, B2, ..., Bn] 
    '''
    simulated_data = pd.DataFrame(X_data.sample(n=data_size, replace=replacement))
    simulated_data.reset_index(inplace = True, drop = True)
    Obs, Lambda = [], []
    coef_vector = np.array(coef_vector)  

    for X in simulated_data.values:

        scale = error_var/error_mean
        shape = error_mean/scale

        error_term = np.random.gamma(shape = shape, scale=scale)     

        ro = np.exp(np.dot(coef_vector,X)+ constant)
        lambda_true = ro * error_term
        Lambda.append(lambda_true)
        Obs.append(np.random.poisson(lam = lambda_true))

    simulated_data['Obs'] = Obs
    simulated_data['Lambda']  = Lambda
    
    return simulated_data
