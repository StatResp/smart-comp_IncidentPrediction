"""
@Author - Ayan Mukhopadhyay
Survival Regression Forecaster -- Inherits from Forecaster class
Implements an uncensored exponential survival model
log T ~ w_0 x_0 + w_1 x_1 + ... w_m x_m + epsilon, where the error term
epsilon is distributed according to the standard extreme-value distribution
"""

from forecasters.base import Forecaster
import numpy as np


def get_likelihood(df, model_params, features):
    w = np.asarray(model_params)
    x = np.array(df[features])
    w_x = w.T @ x.T
    log_t = np.log(np.array(df['time_bet'])).reshape(-1, 1).T
    diff = log_t - w_x
    return (diff - np.exp(diff)).sum()


def _do_gradient_step(df, w, features, alpha=0.01):
    x = np.array(features)
    w_x = w.T @ x.T
    grad = np.zeros((1, w.shape()[1]))
    log_t = np.log(np.array(df['time_bet'])).reshape(-1, 1).T
    diff = np.exp(log_t - w_x)
    for j in range(len(features)):
        x_j = x[:, j]
        grad[1, j] = (-x_j + x_j*diff).sum()

    w += alpha * grad
    return w


def do_gradient_descent(df, features):
    w = np.zeros((1, len(features)))
    l_old = -1e10
    l = get_likelihood(df, model_params=w, features=features)
    tolerance = 10
    while l-l_old > tolerance:
        w_old = w
        l_old = l
        w = _do_gradient_step(df, w, features)
        l = get_likelihood(df, model_params=w, features=features)

    return w_old


class Survival_Model(Forecaster):

    def __init__(self):
        self.name = 'Survival_Regression'
        self.model_params = {}

    def fit(self, df, metadata):
        """
        Fits regression model to data
        @param df: dataframe of incidents in regression format
        @param metadata: dictionary with start and end dates, columns to regress on, cluster labels etc.
                        see github documentation for details
        @return: _
        """
        clusters = df.cluster_label.unique()
        for temp_cluster in clusters:
            df_cluster = df.loc[df.cluster_label == temp_cluster]
            w = do_gradient_descent(df_cluster, metadata['features'])
            self.model_params[temp_cluster] = list(w)

        print('Finished Learning {} model'.format(self.name))

    def predict(self, x_test):
        """
        Predicts E(y|x) for a set of x, where y is the concerned dependent variable.
        @param x_test: dataframe consisting of x points
        @param metadata: dictionary with start and end dates, columns to regress on, cluster labels etc.
                        see github documentation for details
        @return: dataframe with samples
        """
        pass

    def get_regression_expr(self):
        """
        Creates regression expression in the form of a patsy expression
        @param features: x features to regress on
        @return: string expression
        """
        pass
