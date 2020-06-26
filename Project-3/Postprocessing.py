#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: Accuracy
#######################################################################################################################

""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""

from utils import *
import numpy as np
import copy

def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}

    npps = {}
    tss = []

    # Find NPP for all thresholds and all races
    for threshold in np.arange(0, 1.0, 0.01):
        tss.append(threshold)

        for k, v in categorical_results.items():
            vals = apply_threshold(v, threshold)
            if not k in npps:
                npps[k] = []
            
            npps[k].append(get_num_predicted_positives(vals)/len(v))
    
    # Find NPP common in all races 
    temp = np.array(list(npps.values())[0])
    for k, v in npps.items():
        v = np.array(v)
        temp = v[(np.abs(temp[:,None] - v) < epsilon).any(0)]

    xx = []
    for best_npp in temp:
        for k, v in npps.items():
            tr = tss[(np.abs(v - best_npp)).argmin()]
            thresholds[k] = tr
            demographic_parity_data[k] = apply_threshold(categorical_results[k], tr)

        cost = apply_financials(demographic_parity_data)
        xx.append((cost, copy.deepcopy(demographic_parity_data), copy.deepcopy(thresholds)))


    cost, demographic_parity_data, thresholds = max(xx, key=lambda x : x[0])

    return demographic_parity_data, thresholds


#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):
    thresholds = {}
    equal_opportunity_data = {}
    

    tprs = {}
    tss = []

    # Find TPR for all thresholds and all races
    for threshold in np.arange(0, 1.0, 0.01):
        tss.append(threshold)
        for k, v in categorical_results.items():
            vals = apply_threshold(v, threshold)

            if not k in tprs:
                tprs[k] = []

            tprs[k].append(get_true_positive_rate(vals))

    
    # Find TPR common in all races 
    temp = np.array(list(tprs.values())[0])
    for k, v in tprs.items():
        v = np.array(v)
        temp = v[(np.abs(temp[:,None] - v) < epsilon).any(0)]


    xx = []
    for best_tpr in temp:
        for k, v in tprs.items():
            tr = tss[(np.abs(v - best_tpr)).argmin()]
            thresholds[k] = tr
            equal_opportunity_data[k] = apply_threshold(categorical_results[k], tr)

        cost = apply_financials(equal_opportunity_data)
        xx.append((cost, copy.deepcopy(equal_opportunity_data), copy.deepcopy(thresholds)))

    # print('xx11', xx)
    cost, equal_opportunity_data, thresholds = max(xx, key=lambda x : x[0])

    return equal_opportunity_data, thresholds


#######################################################################################################################
"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}

    macc = {}
    
    # Find TPR for all thresholds and all races
    for threshold in np.arange(0, 1.0, 0.005):

        for k, v in categorical_results.items():
            vals = apply_threshold(v, threshold)
            
            if not k in mp_data:
                mp_data[k] = vals
                thresholds[k] = threshold
                macc[k] = apply_financials(vals, True) / len(v)

            acc = apply_financials(vals, True) / len(v)
            if acc > macc[k]:
                mp_data[k] = vals
                macc[k] = acc
                thresholds[k] = threshold
            

    return mp_data, thresholds


#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}

    ppvs = {}
    pvs = []


    # Find TPR for all thresholds and all races
    for threshold in np.arange(0, 1.0, 0.01):
        pvs.append(threshold)
        for k, v in categorical_results.items():
            vals = apply_threshold(v, threshold)

            if not k in ppvs:
                ppvs[k] = []

            ppvs[k].append(get_positive_predictive_value(vals))

    
    # Find TPR common in all races 
    temp = np.array(list(ppvs.values())[0])
    for k, v in ppvs.items():
        v = np.array(v)
        temp = v[(np.abs(temp[:,None] - v) < epsilon).any(0)]

    xx = []
    for best_ppv in temp:
        for k, v in ppvs.items():
            pv = pvs[(np.abs(v - best_ppv)).argmin()]
            thresholds[k] = pv
            predictive_parity_data[k] = apply_threshold(categorical_results[k], pv)

        cost = apply_financials(predictive_parity_data)
        xx.append((cost, copy.deepcopy(predictive_parity_data), copy.deepcopy(thresholds)))


    cost, predictive_parity_data, thresholds = max(xx, key=lambda x : x[0])

    return predictive_parity_data, thresholds


    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}

    xx = []
    for threshold in np.arange(0, 1.0, 0.01):
        for k, v in categorical_results.items():
            vals = apply_threshold(v, threshold)
            thresholds[k] = threshold
            single_threshold_data[k] = apply_threshold(v, threshold)

        cost = apply_financials(single_threshold_data)
        xx.append((cost, copy.deepcopy(single_threshold_data), copy.deepcopy(thresholds)))
            

    # print('xx', xx)
    cost, single_threshold_data, thresholds = max(xx, key=lambda x : x[0])

    return single_threshold_data, thresholds
