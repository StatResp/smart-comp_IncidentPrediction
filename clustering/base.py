"""
@Author - Ayan Mukhopadhyay
Base Class for Spatial-Temporal Clustering. All clustering child classes must implement -

1. similarity_measure - a definition of the distance used to check difference between two units. typically a norm
2. setup_learning_model - for learning dependent clustering methods, setup a regression model
3. merge - merge existing clusters
4. fit - fit clusters to given data
"""


class Clustering_Algo():

    def __init__(self):
        self.name = None

    def similarity_measure(self, x_i, x_j, norm=None):
        pass

    def setup_learning_model(self):
        pass

    def merge(self, df, df_clusters, label_field):
        pass

    def fit(self, df, label_field=None):
        pass
