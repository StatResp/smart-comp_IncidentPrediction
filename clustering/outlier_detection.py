"""
@Author: Ayan Mukhopadhyay
Outlier detection -- Given a cluster of segments, and a forecasting method,
this file applies statistical tests to detect outliers within the cluster in terms of incident arrival.
"""
import numpy as np
import pandas as pd
from copy import deepcopy
import scipy.stats as stats
from math import ceil
from forecasters.regression_models import learn


def detect_outliers(df, cluster_dict, forecasting_model, metadata=None, alpha=None, outlier_fraction=None):
    """
    Detect outliers from a given clustering model
    @param df: dataframe of incidents in regression format
    @param cluster_dict: dictionary with cluster id --> cluster members
    @param forecasting_model: forecasting model. If string is supplied, a model is learned. Otherwise the model is used.
    @param metadata: metadata with unit name
    @param alpha: alpha to be used for ESD Test
    @param outlier_fraction: max fraction of outliers
    @return: dictionary with cluster id --> outlier
    """
    # set default values
    if alpha is None:
        alpha = 0.95
    if outlier_fraction is None:
        outlier_fraction = 0.01
    if metadata is None:
        metadata = {'unit_name': 'unit_segment_id'}

    # if learned model is not supplied, learn one
    if isinstance(forecasting_model, str):
        forecasting_model, reg_df = learn(df=df, metadata=deepcopy(metadata), model_name=forecasting_model,  return_reg_df=True)

    outliers_dict = {cluster_id: [] for cluster_id in cluster_dict.keys()}
    mean_dens = []
    units = []
    for cluster_id, members in cluster_dict.items():
        # get forecasting mean densities
        for member_id in members:
            likelihood = forecasting_model.get_likelihood(reg_df.loc[reg_df[metadata['unit_name']] == member_id],  metadata=metadata)
            units.append(member_id)
            mean_dens.append(likelihood)
        outlier_temp = esd_Test(mean_dens, alpha=alpha, max_outliers=int(ceil(outlier_fraction*len(mean_dens))))
        outliers_dict[cluster_id] = outlier_temp

    return outliers_dict


def grubbs_stat(y):
    std_dev = np.std(y)
    avg_y = np.mean(y)
    abs_val_minus_avg = abs(y - avg_y)
    max_of_deviations = max(abs_val_minus_avg)
    max_ind = np.argmax(abs_val_minus_avg)
    g_cal = max_of_deviations/std_dev
    return g_cal, max_ind


def critical_value(size, alpha):
    t_dist = stats.t.ppf(1 - alpha / (2 * size), size - 2)
    numerator = (size - 1) * np.sqrt(np.square(t_dist))
    denominator = np.sqrt(size) * np.sqrt(size - 2 + np.square(t_dist))
    cv = numerator / denominator
    return cv


def check_G_values(g_stat, gc, inp, max_index):
    if g_stat > gc:
        return True
    return False


def esd_Test(input, alpha, max_outliers):
    outliers = []
    for i in range(max_outliers):
        gc = critical_value(len(input), alpha)
        g_stat, max_index = grubbs_stat(input_series)
        if check_G_values(g_stat, gc, input_series, max_index):
            outliers.append(input[i])
        input_series = np.delete(input_series, max_index)

    return outliers



