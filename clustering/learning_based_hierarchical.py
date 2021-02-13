"""
@Author - Ayan Mukhopadhyay
Learning based Hierarchical Clustering Model -- Same as a hierarchical
clustering model, but it takes as input a forecasting model and merges
clusters only if likelihood increases on unseen data.
"""

from clustering.base import Clustering_Algo
from numpy.linalg import norm
import pandas as pd
import numpy as np
from forecasters.poisson_reg_forecaster import GLM_Model
from copy import deepcopy
from clustering.utils import *
from forecasters.utils import create_regression_df
import sys


def update_map(df, map, metadata):
    """
    Update df's units based on map, which is a mapping between original and new mapping
    If there is no mapping in the original map, put None.
    """

    def apply_func(x):
        try:
            new_unit = map[x[metadata['unit_name']]]
        except KeyError:
            new_unit = None

        return new_unit

    df[metadata['unit_name']] = df.apply(apply_func, axis=1)
    return df


def transform_metadata(metadata, df_train, df_test):
    # sanity steps for the clustering unit
    if metadata['unit_name'] is None:
        metadata['unit_name'] = 'cell'
    metadata_test = metadata
    metadata_train = metadata
    metadata_test['start_time'] = df_test.time.min()
    metadata_test['end_time'] = df_test.time.max()
    metadata_train['start_time'] = df_train.time.min()
    metadata_train['end_time'] = df_train.time.max()
    return metadata, metadata_train, metadata_test


class Hierarchical_Cluster_L(Clustering_Algo):

    def __init__(self, learning_model_name=None):
        self.name = "Hierarchical Clustering"
        self.learning_model_name = learning_model_name
        self.learning_model = None

    def similarity_measure(self, x_i, x_j, norm_order=None):
        """
        Similarity measure for clustering
        @param x_i: unit i features
        @param x_j: unit j features
        @param norm_order: type of norm
        @return: distance score
        """
        return norm(x_i - x_j, norm_order)

    def setup_learning_model(self, df_):
        if self.learning_model_name == 'Poisson_Regression':
            self.learning_model = Poisson_Model()

        split_point = 0.8
        mask = np.random.rand(len(df_)) < split_point
        df_train = df_[mask].reset_index()
        df_test = df_[~mask].reset_index()
        # Add Intercept term to df_test --> used in prediction
        return df_train, df_test

    def merge(self, df_, groups_, i_, j_, col):
        """
        Merge two existing clusters
        @param df_: dataframe with cluster information
        @param groups_: dictionary that map clusters to rows in incident data
        @param i_: cluster id 1
        @param j_: cluster id 2
        @param col: columns to use for clustering (the set of static featurs)
        @return: updated dataframe and groups dict with new clusters post merging
        """
        # merge the closest clusters x_i and x_j into x_j & remove x_i
        df_.loc[df_.cluster_label == i_]['cluster_label'] = j_
        df_.iloc[j_][col] = df_.iloc[[i_, j_]][col].mean(axis=0)
        df_.drop(df_.index[[i_]])
        # update the groups dict
        groups_[j_].append(groups_[i_])
        groups_.pop(i_)
        return df_, groups_

    def learn_pred_cluster(self, df, df_test, meta_train, meta_test):
        self.learning_model.fit(df, meta_train)
        lik = self.learning_model.get_likelihood(df_test, meta_test)
        return lik

    def fit(self, df, metadata=None, norm_order=None, stop_at=None):
        """
        fit clustering model to data
        @param df: dataframe with incidents
        @param metadata: dictionary with static features, unit name to cluster on,
        @param norm_order: norm order for similarity measure
        @param stop_at: min number of acceptable clusters
        @return: dictionary mapping clusters to data
        """
        # metadata must be supplied
        if metadata['static_features'] is None:
            print("A list of static features must be supplied to learn clusters")
            sys.exit()
        else:
            temp_columns = deepcopy(metadata['static_features'])
        temp_columns.append('cluster_label')

        # create a forecasting model and keep test data away
        df_train, df_test = self.setup_learning_model(df)

        # update metadata params based on train and test sets
        metadata, metadata_train, metadata_test = transform_metadata(metadata, df_train, df_test)

        df[metadata['unit_name']] = df[metadata['unit_name']].astype('int64')

        # update test data with transformed unit. Remove units that are not present in training
        df_test = df_test[df_test[metadata['unit_name']] != None]

        # Initialize each cell as its own cluster & create dict to track clusters
        df_cluster, groups = init_clusters(df_train, metadata)
        # update df_train and df_test with cluster assignment
        df_train['cluster_label'] = df_train.apply(lambda x: [k for k in list(groups.keys()) if x[metadata['unit_name']]
                                                              in groups[k]][0], axis=1)
        df_test['cluster_label'] = df_test.apply(lambda x: [k for k in list(groups.keys()) if x[metadata['unit_name']]
                                                            in groups[k]][0], axis=1)

        # create regression dataframe - get counts per time step
        regression_df = create_regression_df(df_train, metadata_train, self.learning_model_name)
        regression_df_test = create_regression_df(df_test, metadata_test, self.learning_model_name)
        regression_df_test['Intercept'] = 1

        # initialize cluster dataframe
        df_cluster = df_cluster[temp_columns]

        # merge till learning improves
        old_lik = self.learn_pred_cluster(df=regression_df, df_test=regression_df_test, meta_train=metadata_train,
                                          meta_test=metadata_test)
        learn_improve = True
        while learn_improve:
            # get candidate clusters
            df_cluster.reset_index(drop=True, inplace=True)
            cand_i, cand_j = find_closest(df_cluster, metadata['static_features'], self.similarity_measure, norm_order)

            # check if learning improves by merging. keep temporary copies of train and test dataframes
            reg_temp = deepcopy(regression_df)
            reg_temp['cluster_label'] = reg_temp.apply(
                lambda x: cand_j if x['cluster_label'] == cand_i else x['cluster_label'], axis=1)
            reg_temp_test = deepcopy(regression_df_test)
            reg_temp_test['cluster_label'] = reg_temp_test.apply(
                lambda x: cand_j if x['cluster_label'] == cand_i else x['cluster_label'], axis=1)

            curr_lik = self.learn_pred_cluster(df=reg_temp, df_test=reg_temp_test, meta_train=metadata_train,
                                               meta_test=metadata_test)

            if curr_lik > old_lik:
                df_cluster, groups = self.merge(df_cluster, groups, cand_i, cand_j,
                                                metadata['static_features'])
                continue
            # if not, break loop
            learn_improve = False

        return df_cluster
