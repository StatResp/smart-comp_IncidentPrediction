"""
@Author - Ayan Mukhopadhyay
Utility methods for clustering algorithms
"""
from itertools import combinations


def create_unit_map(df, units, unit_field):
    """Creates a map between "n" units with arbitrary int/float identifiers to ints between 0 and n-1"""
    df[unit_field] = df[unit_field].astype('int64')
    unit_map = {units[i]: i for i in range(len(units))}
    df[unit_field] = df.apply(lambda x: unit_map[x[unit_field]], axis=1)
    return df, unit_map


def init_clusters(df, meta):
    # Initialize each cell as its own cluster & create dict to track clusters
    gp_obj = df.groupby(meta['unit_name'])
    df_cluster = gp_obj[meta['static_features']].mean()
    df_cluster['cluster_label'] = df_cluster.index
    # create groups
    groups_dict = {}
    for k, val in gp_obj.groups.items():
        temp = []
        for ind in list(val):
            try:
                temp.append(df.iloc[ind][meta['unit_name']])
            except IndexError as e:
                print("Error row in dataframe")
                raise e

        # get rid of duplicates but keep as a list to aid appending multiple times later
        groups_dict[k] = list(set(temp))
    return df_cluster, groups_dict


def find_closest(df_cluster, static_features, similarity_measure, norm_order):
    # identify the closest clusters
    val = df_cluster[static_features].values
    min_norm = 1e10
    curr_indices = (0, 0)
    for t in combinations(df_cluster.index, 2):
        temp_dist = similarity_measure(val[int(t[0])], val[int(t[1])], norm_order=norm_order)
        if temp_dist < min_norm:
            min_norm = temp_dist
            curr_indices = t

    return df_cluster.iloc[curr_indices[0]]['cluster_label'], df_cluster.iloc[curr_indices[1]]['cluster_label']


def reset_keys(dictionary):
    i = 0
    reset_dict = {}
    for k, val in dictionary.items():
        reset_dict[i] = val
        i += 1
    return reset_dict
