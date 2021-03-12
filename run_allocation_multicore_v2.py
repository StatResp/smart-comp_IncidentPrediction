# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:51:40 2021

@author: Sayyed Mohsen Vazirizade
"""
import time
import pickle5 as pickle
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

from multiprocessing import Pool
import multiprocessing
import json

import os.path
from os import path

#%%

def get_allocation(arguments):
    '''

    '''

    num_resources, alpha, model_i, DF_Test_spacetime, All_seg_incident, Grid_center, df_incident, Delay, Speed, Penalty, Window_size_hour, possible_facility_locations, demand_nodes, Distant_Dic, time_range = arguments
    print(num_resources, alpha, model_i, Delay, Penalty, Window_size_hour)
    df_responders_Exist=False
    Figure_Tag=False
    All_Responders_GridID={}
    if path.exists('results/ResponderLocation/Responder_Location_@'+model_i+'_'+str(num_resources)+'V'+str(Delay)+'h'+str(Speed)+'S'+str(alpha)+'alpha.json')==True:
        print('Exist')    
    else:
        for row,row_values  in time_range.iloc[0:2186].iteritems(): 
                if (model_i!='Naive') | (df_responders_Exist==False) | (len(time_range)>186):
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
                    df_responders_Exist=True
                    #print(df_responders[['ID','Grid_ID']])
                else:
                    print(row, row_values, 'for naive model df_responders just generated once')
                #All_Responders_GridID.append(list(Responders_GridID))
                All_Responders_GridID[str(row)]=list(Responders_GridID)
        with open('results/ResponderLocation/Responder_Location_@'+model_i+'_'+str(num_resources)+'V'+str(Delay)+'h'+str(Speed)+'S'+str(alpha)+'alpha.json', "w") as f:    
            json.dump(All_Responders_GridID, f, indent = 6) 

def get_distance(arguments):
    '''

    '''
    num_resources, alpha, model_i, DF_Test_spacetime, All_seg_incident, Grid_center, df_incident, Delay, Speed, Penalty, Window_size_hour, possible_facility_locations, demand_nodes, Distant_Dic, time_range = arguments
    print(num_resources, alpha, model_i, Delay, Penalty, Window_size_hour)
    DF_metric=pd.DataFrame({'time_local':time_range})
    df_responders_Exist=False
    Figure_Tag=False
    with open('results/ResponderLocation/Responder_Location_@'+model_i+'_'+str(num_resources)+'V'+str(Delay)+'h'+str(Speed)+'S'+str(alpha)+'alpha.json') as f:
        All_Responders_GridID = json.load(f)      
    for row,row_values  in time_range.iloc[0:2186].iteritems(): 
        if (model_i!='Naive') | (df_responders_Exist==False) | (len(time_range)>186):
            print(row,row_values)
            weights_dict, DF_Test_space_time_i=Weight_and_Merge(DF_Test_spacetime,All_seg_incident,time_i=row_values,model=model_i)
            Responders_GridID=All_Responders_GridID[str(row)]
            df_responders=Responders_Location(Grid_center,Responders_GridID,DF_Test_space_time_i,time_i=row_values,model=model_i, alpha=alpha, Figure_Tag=False)
            df_responders_Exist=True
            #print(df_responders[['ID','Grid_ID']])
        else:
            print(row, row_values, 'for naive model df_responders just generated once')

        Dispatch_Output=Dispaching_Scoring(row_values,df_responders,df_incident ,Window_size_hour,Delay,Speed,Penalty,model_i=model_i,alpha=alpha, Figure_Tag=False)
        DF_metric.loc[row,list(Dispatch_Output.keys())]=list(Dispatch_Output.values())
        Figure_Tag=False
    #DF_metric.mean().to_json('results/Distance/Distance-Metric_@'+model_i+'_'+str(num_resources)+'V'+str(Delay)+'h'+str(Speed)+'S'+str(alpha)+'alpha.json',orient='table')
    DF_metric.to_json('results/Distance/Distance-Metric_@'+model_i+'_'+str(num_resources)+'V'+str(Delay)+'h'+str(Speed)+'S'+str(alpha)+'alpha.json',orient='table')




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

if __name__ == "__main__":

    #DF_Test_spacetime=pd.read_pickle('results/DF_Test_spacetime_All.pkl')     #pd.read_pickle('D:/inrix/prediction_engine_20/results/DF_Test_spacetime_AllMethods_None_Alltestwindow.pkl')
    DF_Test_spacetime=pickle.load(open('results/DF_Test_spacetime_All.pkl', 'rb')) 
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
    num_resources_list =[10,15,20] #[2, 3]# [10,15,20]
    alpha_list=[0.0,0.5,1.0,2.0]  #[0,1]#[0.0,0.5,1.0,2.0]
    #model_list= ['Naive','LR+NoR+NoC1','RF+CW+NoC1'] 
    #model_list=['Naive','LR+NoR+NoC1', 'LR+RUS+NoC1','LR+ROS+NoC1','LR+NoR+KM2','LR+RUS+KM2','LR+ROS+KM2']                  
    #model_list=['RF+CW+NoC1', 'RF+NoR+NoC1', 'RF+RUS+NoC1','RF+ROS+NoC1','RF+CW+KM2', 'RF+NoR+KM2', 'RF+RUS+KM2','RF+ROS+KM2']
    #model_list=['NN+NoR+NoC1', 'NN+RUS+NoC1','NN+ROS+NoC1','NN+NoR+KM2','NN+RUS+KM2','NN+ROS+KM2'] 
    #All
    
    model_list=['Naive','LR+NoR+NoC1', 'LR+RUS+NoC1','LR+ROS+NoC1','LR+NoR+KM2','LR+RUS+KM2','LR+ROS+KM2',
                'RF+CW+NoC1', 'RF+NoR+NoC1', 'RF+RUS+NoC1','RF+ROS+NoC1','RF+CW+KM2', 'RF+NoR+KM2', 'RF+RUS+KM2','RF+ROS+KM2',
                'NN+NoR+NoC1', 'NN+RUS+NoC1','NN+ROS+NoC1','NN+NoR+KM2','NN+RUS+KM2','NN+ROS+KM2',
                'ZIP+NoR+NoC1','ZIP+RUS+NoC1','ZIP+ROS+NoC1','ZIP+NoR+KM2','ZIP+RUS+KM2','ZIP+ROS+KM2']
    
    
    
    #model_list=['ZIP+NoR+NoC1','ZIP+RUS+NoC1','ZIP+ROS+NoC1','ZIP+NoR+KM2','ZIP+RUS+KM2','ZIP+ROS+KM2']
    Delay=0.5
    Speed=100
    Penalty=np.nan
    Window_size_hour=4

    HyperParams=[]
    row_id=0
    for num_resources in num_resources_list:
        for alpha in alpha_list:
            for model_i in model_list:
                HyperParams.append([num_resources,alpha,model_i])

    experimental_inputs = []
    for num_resources, alpha, model_i in HyperParams:
        # print(num_resources, alpha, model_i)

        input_array = [num_resources,
                       alpha,
                       model_i,
                       DF_Test_spacetime,
                       All_seg_incident,
                       Grid_center,
                       df_incident,
                       Delay,
                       Speed,
                       Penalty,
                       Window_size_hour,
                       possible_facility_locations,
                       demand_nodes,
                       Distant_Dic,
                       time_range]

        experimental_inputs.append(input_array)

    print('starting experiments')
    start_time = time.time()
    # get_dist_metric_for_allocation(experimental_inputs[0])
    with Pool(processes=35) as pool:
        res_dict = pool.map(get_allocation, experimental_inputs)

    print('computation time for allocation: {}'.format(time.time() - start_time))
    print('starting experiments')
    start_time = time.time()
    with Pool(processes=35) as pool:
        res_dict = pool.map(get_distance, experimental_inputs)

    # for args in experimental_inputs:
    #     get_dist_metric_for_allocation(args)


    print('computation time for distance: {}'.format(time.time() - start_time))









