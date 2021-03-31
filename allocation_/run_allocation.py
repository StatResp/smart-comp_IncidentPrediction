# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:51:40 2021

@author: vaziris
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pygeoj
import pyproj
from allocation.Griding_TN import MyGrouping_Grid, Distance_Dict_Builder,Distance_Dict_Builder
from allocation.pmedianAllocator import pmedianAllocator
from allocation.Allocation import Dispaching_Scoring, Find_Best_Location, Weight_and_Merge, Responders_Location, Graph_Distance_Metric

import pickle5 as pickle




#%%Input
'''
metadata
#NAME='DF_Test_metric_time_AllMethods_None_Alltestwindow'
NAME='DF_Test_metric_time_LR'
#1)
DF_Test_metric_time=pd.read_pickle('output/'+NAME+'.pkl')   #pd.read_pickle('D:/inrix/prediction_engine_20/output/DF_Test_metric_time_AllMethods_None_Alltestwindow.pkl')
'''
#2
DF_Test_spacetime=pickle.load(open('output/DF_Test_spacetime_All.pkl', 'rb'))     #pd.read_pickle('D:/inrix/prediction_engine_20/output/DF_Test_spacetime_AllMethods_None_Alltestwindow.pkl')

df_=DF_Test_spacetime[['XDSegID']] #df_=pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/Regression_4h_History_G_top20_NoNA.pkl')
#3)
df_grouped =pd.read_pickle('sample_data/data/cleaned/Line/grouped/grouped_3.pkl')#pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/grouped/grouped_3.pkl')
All_seg_incident=df_grouped[df_grouped['Grouping'].isin(df_['XDSegID'])].reset_index().drop('index',axis=1)
#4)
df_inrix=pd.read_pickle('sample_data/data/cleaned/Line/inrix_grouped.pkl')#pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/inrix_grouped.pkl')
#5)
df_incident =pd.read_pickle('sample_data/data/cleaned/Line/incident_XDSegID.pkl')#pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/incident_XDSegID.pkl')
df_incident =df_incident.sort_values('time_local')

time_range=DF_Test_spacetime['time_local'].drop_duplicates().sort_values()
#%%Griding
possible_facility_locations,demand_nodes,Distant_Dic,All_seg_incident,Grid_center=Find_Best_Location(df_inrix,df_incident,All_seg_incident,width = 0.1,height = 0.1  )


#%%Features
num_resources_list =[2,3]# [10,15,20]
alpha_list=[0,1]#[0,0.5,1,2]
model_list= ['naive','LR+NoR+NoC1','RF+CW+NoC1'] 
                #['naive','LR+NoR+NoC1', 'LR+RUS+NoC1','LR+ROS+NoC1','LR+NoR+KM2','LR+RUS+KM2','LR+ROS+KM2']                   #  'Logistic_Regression+No_Resample'#'Logistic_Regression+No_Resample'  #model_i='Naive'
                # ['RF+CW+NoC1', 'RF+NoR+NoC1', 'RF+RUS+NoC1','RF+ROS+NoC1','RF+CW+KM2', 'RF+NoR+KM2', 'RF+RUS+KM2','RF+ROS+KM2']
Delay=1
Penalty=np.nan
Window_size_hour=4

HyperParams=[]
row_id=0
for num_resources in num_resources_list:
    for alpha in alpha_list:
        for model_i in model_list:
            HyperParams.append([num_resources,alpha,model_i])



#%%Run

'''
HyperParams
DF_Test_spacetime
All_seg_incident
Grid_center
Responders_GridID
DF_Test_space_time_i
df_incident

Delay, Penalty, Window_size_hour
'''



for counter, [num_resources, alpha, model_i] in enumerate(HyperParams):
    print(counter, num_resources, alpha,model_i , model_i)
    DF_metric=pd.DataFrame({'time_local':time_range})
    df_responders_Exist=False
    Figure_Tag=False
    for row,row_values  in time_range.iloc[0:2].iteritems(): 
        if (model_i!='naive') | (df_responders_Exist==False):
            print(row,row_values)
            weights_dict, DF_Test_space_time_i=Weight_and_Merge(DF_Test_spacetime,All_seg_incident,time_i=row_values,model=model_i)
            allocator = pmedianAllocator()
            Responders_GridID = allocator.solve(number_of_resources_to_place=num_resources,
                                         possible_facility_locations= possible_facility_locations,
                                         demand_nodes=demand_nodes,
                                         distance_dict=Distant_Dic,
                                         demand_weights=weights_dict,
                                         score_type='penalty',
                                         alpha=alpha)
            df_responders=Responders_Location(Grid_center,Responders_GridID,DF_Test_space_time_i,time_i=row_values,model=model_i, alpha=alpha, Figure_Tag=False)
            df_responders_Exist=True
            #print(df_responders[['ID','Grid_ID']])
        else:
            print(row, row_values, 'for naive model df_responders just generated once')
        Dispatch_Output=Dispaching_Scoring(row_values,df_responders,df_incident ,Window_size_hour,Delay,Penalty,model_i=model_i,alpha=alpha, Figure_Tag=False)
        DF_metric.loc[row,list(Dispatch_Output.keys())]=list(Dispatch_Output.values())
        Figure_Tag=False
    DF_metric.mean().to_json('output/Distance/Distance-Metric_@'+model_i+'_'+str(num_resources)+'V'+str(Delay)+'h'+str(alpha)+'alpha.json',orient='table')











