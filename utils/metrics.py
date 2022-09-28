
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Mohammad Zarei
# Created Date: 30 Dec 2019
# ---------------------------------------------------------------------------
"""Metrics used to evaluate performance of NB models (SPFs)"""
# ---------------------------------------------------------------------------
# Imports

import pandas as pd
import numpy as np




def MAPE(actual, predicted):
    '''
    Returns Mean Absolute Percentage Error.

    actual, predicted: numpy arrays 
    '''
    res = np.empty(len(actual))
    for j in range(len(actual)):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return np.mean(np.abs(res))



def FI(results):
    '''
    Returns average of False Identification Test for top n% hotspots where n is 2,4,6,8,10.
    results dataframe must have 'true_rank' and 'rank' columns
    '''

    FI_test = 0
    for HS_level in [0.02,0.04,0.06,0.08,0.1]:
        FI_test += 1-results[(results['true_rank'] > 1- HS_level) & (results['rank'] > 1- HS_level)].count()[0]/results[(results['rank'] > 1- HS_level)].count()[0]
    return round(FI_test/5, 3)


def PMD(results):
    '''
    Returns average of Poison Mean Different Test for top n% hotspots where n is 2,4,6,8,10.
    results dataframe must have 'true_rank', 'rank', 'lambda' columns
    '''
    PMD_test = 0
    for HS_level in [0.02,0.04,0.06,0.08,0.1]:
        PMD_test += (results[results['true_rank'] > 1-HS_level]['lambda'].mean() - results[results['rank'] > 1-HS_level]['lambda'].mean())/results[results['true_rank'] > 1-HS_level]['lambda'].mean()
    return round(PMD_test/5,3)



def CURE(df, ax, x_label, plot_label, ls='-', y_name='Obs', x='Ftot', y_pred='y_pred',boundry=True):
    '''
    Plot Cumulative Residual plot based on predictions ('y_pred') and observed crash ('Obs') counts in 'df'
    '''
    df.sort_values(by=x, ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['i'] = range(1, len(df[y_name])+1)
    df['res'] = df[y_pred] - df[y_name]
    df['res_sq'] = (df[y_pred] - df[y_name])**2

    df['e1'] = df['res'].cumsum()
    
    df.iloc[-1, df.columns.get_loc('e1')] = 0

    df['e2'] = df['res_sq'].cumsum()
    df['e3'] = (df['e2']*((1 - df['e2']/df['e2'].iloc[-1])))**0.5
    area_out_of_bound = len(df[(df.e1 > 2*df.e3) | (df.e1 < -2*df.e3)])/len(df)
    print(area_out_of_bound)
    if boundry == True:
        ax.plot(df[x],  2.5*df['e3'], linestyle = 'dotted',label="2$\sigma$")
        ax.plot(df[x], -2.5*df['e3'], linestyle = 'dotted',label="-2$\sigma$")
    ax.plot(df[x], df['e1'], linestyle = ls, label = plot_label)
    ax.legend(loc='upper right',fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel('CURE')
    return df
