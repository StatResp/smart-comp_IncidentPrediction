# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:26:26 2020

@author: Sayyed Mohsen Vazirizade
"""

import numpy as np
import pandas as pd
from patsy import dmatrices
from copy import deepcopy
from scipy.special import factorial
from scipy import stats
from scipy.special import gamma
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE



def Balance_Adjuster(df,metadata):
    """
    caclulate the Balance ratio between each cluster for resampling
    @param df: the big regression dataframe
    @param metadata:
    @return BalanceFactor: it is a list of balanced ratio for each cluster sorted by cluster number
    """

    
    Sparsity=df[['cluster_label',metadata['pred_name_TF']]].groupby('cluster_label').mean().reset_index().sort_values('cluster_label',ascending=True )[metadata['pred_name_TF']].tolist()
    BalanceFactor=[i/max(Sparsity) for i in Sparsity]
    return BalanceFactor
    
    




def Resampling_Func(sampling_name,BalanceFactor):
    """
    choose the resampling method
    @param sampling_name: string that explains the resampling method
    @param BalanceFactor: the BalanceFactor for one indivudual cluster
    @return sampling:it is the model for resampling
    """   
    
    if sampling_name in ['RUS','ROS','SMOTE']:
        if sampling_name=='RUS':        
            
            sampling = RandomUnderSampler(sampling_strategy=BalanceFactor,random_state=1)
        elif sampling_name=='ROS':        
            
            sampling = RandomOverSampler(sampling_strategy=BalanceFactor,random_state=1)
        elif sampling_name=='SMOTE':

            #from imblearn.over_sampling import BorderlineSMOTE
            sampling = SMOTE(sampling_strategy=BalanceFactor,random_state=1)
            #oversample = BorderlineSMOTE()
    return sampling