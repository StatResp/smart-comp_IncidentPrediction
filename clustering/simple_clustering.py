"""
@Author - Ayan Mukhopadhyay
Hierarchical Clustering Model -- Iteratively checks for similarty in the
feature space and merges clusters up to a pre-specified number of clusters
"""

from clustering.base import Clustering_Algo
from numpy.linalg import norm
from itertools import combinations
import pandas as pd
#from forecasters import poisson_reg_forecaster
from clustering.utils import *
from copy import deepcopy
import sys
from sklearn.cluster import KMeans

class Simple_Cluster:
    def __init__(self,model_name,n_clusters):
        self.model_name = model_name
        #self.name = "Simple Clustering"
        if (self.model_name == 'kmeans'):
            model = KMeans(n_clusters=n_clusters, random_state=0)
        else:
            model=np.nan
            
            
        self.modelparam=model
            
    
        
    def fit_pred(self, df_train,df_test,df_predict, metadata=None):
        #fit clustering model to data
        #@param df_train: dataframe which used to count the number of accidents for each segment and use it for clustering training
        #@param df_test and df_predict: dataframe we want to apply clustering on it 
        #@param metadata: 
        #@return: a list indicates the cluster id
        gp_obj = df_train.groupby(metadata['unit_name'])
        df_cluster = gp_obj[metadata['pred_name_TF']].mean()
        df_cluster = df_cluster.reset_index()

        if (self.model_name == 'kmeans'):
            df_cluster['extra']=0  #since KMeans needs at least 2 dimensions, we add a fake dimension.
            self.modelparam = self.modelparam.fit(df_cluster[['extra',metadata['pred_name_TF']]])
            df_cluster['cluster_label']=self.modelparam.predict(df_cluster[['extra',metadata['pred_name_TF']]])
            df_train= pd.merge(df_train, df_cluster[['cluster_label', metadata['unit_name']]], left_on=metadata['unit_name'], right_on=metadata['unit_name'], how='left')
            df_test= pd.merge(df_test, df_cluster[['cluster_label', metadata['unit_name']]], left_on=metadata['unit_name'], right_on=metadata['unit_name'], how='left')
            df_predict= pd.merge(df_predict, df_cluster[['cluster_label', metadata['unit_name']]], left_on=metadata['unit_name'], right_on=metadata['unit_name'], how='left')
            print(df_cluster[['cluster_label','extra']].groupby('cluster_label').count())
            print(df_predict[['cluster_label',metadata['pred_name_TF']]].groupby('cluster_label').mean())
        
        return df_train, df_test, df_predict


'''
df_cluster=DF_Regression[['XDSegID','Total_Number_Incidents_average_per_seg','Intercept']].drop_duplicates()

from clustering.base import Clustering_Algo
from numpy.linalg import norm
from itertools import combinations
import pandas as pd
#from forecasters import poisson_reg_forecaster
from clustering.utils import *
from copy import deepcopy
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

modelparam = KMeans(n_clusters=5, random_state=0)
modelparam = modelparam.fit(df_cluster[['Intercept','Total_Number_Incidents_average_per_seg']])
df_cluster['cluster_label']=modelparam.predict(df_cluster[['Intercept','Total_Number_Incidents_average_per_seg']])
print(df_cluster.groupby('cluster_label').count())
print(df_cluster.groupby('cluster_label').max())    
print(df_cluster.groupby('cluster_label').mean())    
print(df_cluster.groupby('cluster_label').min())       


df_cluster['Total_Number_Incidents_average_per_seg'].hist(bins = 50)
for i in df_cluster['cluster_label'].unique().tolist():
    plt.axvline(x=df_cluster[df_cluster['cluster_label']==i]['Total_Number_Incidents_average_per_seg'].mean(),color='Red')
    
    
    
plt.axvline(x=df_cluster[df_cluster['cluster_label']==1]['Total_Number_Incidents_average_per_seg'].mean(),color='Red')
plt.axvline(x=df_cluster[df_cluster['cluster_label']==2]['Total_Number_Incidents_average_per_seg'].mean(),color='Red')    
plt.axvline(x=df_cluster[df_cluster['cluster_label']==3]['Total_Number_Incidents_average_per_seg'].mean(),color='Red')    
plt.axvline(x=df_cluster[df_cluster['cluster_label']==4]['Total_Number_Incidents_average_per_seg'].mean(),color='Red')    
'''