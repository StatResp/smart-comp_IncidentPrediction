"""
@Author - Ayan Mukhopadhyay
Hierarchical Clustering Model -- Iteratively checks for similarty in the
feature space and merges clusters up to a pre-specified number of clusters
"""

from clustering.base import Clustering_Algo
from numpy.linalg import norm
from itertools import combinations
import pandas as pd
from forecasters import poisson_reg_forecaster
from clustering.utils import *
from copy import deepcopy
import sys


class Hierarchical_Cluster:

    def __init__(self):
        self.name = "Hierarchical Clustering"

    def similarity_measure(self, x_i, x_j, norm_order=None):
        """
        Similarity measure for clustering
        @param x_i: unit i features
        @param x_j: unit j features
        @param norm_order: type of norm
        @return: distance score
        """
        return norm(x_i - x_j, norm_order)

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
        # get index of j-th and i-th clusters
        j_index = list(df_.loc[df_['cluster_label'] == j_].index)[0]
        i_index = list(df_.loc[df_['cluster_label'] == i_].index)[0]
        # replace each column with mean
        for c in col:
            df_.at[j_index, c] = df_.loc[(df_.cluster_label == j_) |
                                                         (df_.cluster_label == i_)][c].mean(axis=0)
        df_.drop(i_index, axis=0, inplace=True)
        # update the groups dict
        groups_[j_].extend(groups_[i_])
        groups_.pop(i_)
        return df_, groups_

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

        # sanity step for the clustering unit and dataframe
        df.reset_index(inplace=True, drop=True)
        if metadata['unit_name'] is None:
            metadata['unit_name'] = 'cell'

        # convert unit column to int, replace n arbitrary unit identifiers with numbers from 0 to n-1. Save the map.
        # df, unit_map = create_unit_map(df, df[unit].unique(), unit_field=unit)
        df[metadata['unit_name']] = df[metadata['unit_name']].astype('int64')

        # Initialize each cell as its own cluster & create dict to track clusters
        df_cluster, groups = init_clusters(df, metadata)
        df['cluster_label'] = df.apply(lambda x: [k for k in list(groups.keys()) if x[metadata['unit_name']]   in groups[k]][0], axis=1)

        # iteratively merge till 'stop_at' number of clusters are attained
        stop_at = 5 if stop_at is None else stop_at
        while len(df_cluster) > stop_at:
            # get candidate clusters
            df_cluster.reset_index(drop=True, inplace=True)
            cand_i, cand_j = find_closest(df_cluster, metadata['static_features'], self.similarity_measure, norm_order)
            df_cluster, groups = self.merge(df_cluster, groups, cand_i, cand_j, metadata['static_features'])
        return reset_keys(groups)
