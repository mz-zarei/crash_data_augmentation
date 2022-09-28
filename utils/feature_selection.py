
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
from nb_model import fitNB



def forward_regression(data, features, y_name='Obs', threshold_in=0.05, verbose=False):
    '''
    Returns significant features in a Negative Binomial forward-stepwise regresion
    given the 'data' and 'features' names
    '''

    X = data[features]
    y =  data[y_name]

    initial_list = []
    included = []
    while True:
        changed=False
        excluded = list(set(features)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:

            model, alpha = fitNB(data, included+[new_column], y_name)
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included

def backward_regression(data, features, y_name='Obs', threshold_out=0.05, verbose=False):
    '''
    Returns significant features in a Negative Binomial backward-stepwise regresion
    given the 'data' and 'features' names
    '''

    X = data[features]
    y =  data[y_name]

    included=list(X.columns)
    while True:
        changed=False
        model, alpha = fitNB(data, included, y_name)
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included
