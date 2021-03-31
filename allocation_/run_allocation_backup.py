# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:51:40 2021

@author: vaziris
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import pygeoj
import pyproj
from allocation.Griding_TN import MyGrouping_Grid, Distance_Dict_Builder,Distance_Dict_Builder
from allocation.pmedianAllocator import pmedianAllocator
from allocation.Allocation import Dispaching_Scoring, Find_Best_Location, Weight_and_Merge, Responders_Location, Graph_Distance_Metric


#%%Features
num_resources_list = [10,15,20]
Delay=1
Penalty=np.nan
model_list=  ['naive','LR+NoR+NoC1', 'LR+RUS+NoC1','LR+ROS+NoC1','LR+NoR+KM2','LR+RUS+KM2','LR+ROS+KM2']                   #  'Logistic_Regression+No_Resample'#'Logistic_Regression+No_Resample'  #model_i='Naive'
Window_size_hour=metadata['window_size']/3600
alpha_list=[0,0.5,1,2]
HyperParams=[]
for num_resources in num_resources_list:
    for alpha in alpha_list:
        for model_i in model_list:
            HyperParams.append([num_resources,alpha,model_i])
len(HyperParams)




#%%Input
metadata
#NAME='DF_Test_metric_time_AllMethods_None_Alltestwindow'
NAME='DF_Test_metric_time_All-Method'
#1)
DF_Test_metric_time=pd.read_pickle('output/'+NAME+'.pkl')   #pd.read_pickle('D:/inrix/prediction_engine_20/output/DF_Test_metric_time_AllMethods_None_Alltestwindow.pkl')
#2)
DF_Test_spacetime=pd.read_pickle('output/DF_Test_spacetime_All-Method.pkl')     #pd.read_pickle('D:/inrix/prediction_engine_20/output/DF_Test_spacetime_AllMethods_None_Alltestwindow.pkl')
df_=DF_Test_spacetime[['XDSegID']] #df_=pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/Regression_4h_History_G_top20_NoNA.pkl')
#3)
df_grouped =pd.read_pickle(metadata['groups_pickle_address'])#pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/grouped/grouped_3.pkl')
All_seg_incident=df_grouped[df_grouped['Grouping'].isin(df_['XDSegID'])].reset_index().drop('index',axis=1) 
#4)
df_inrix=pd.read_pickle(metadata['inrix_pickle_address'])#pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/inrix_grouped.pkl')
#5)
df_incident =pd.read_pickle(metadata['incident_pickle_address'])#pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/incident_XDSegID.pkl')
df_incident =df_incident.sort_values('time_local')  
DF_Test_metric_time=DF_Test_metric_time[['time_local']]
time_range=DF_Test_metric_time['time_local'].drop_duplicates().sort_values()


#%%Griding
possible_facility_locations,demand_nodes,Distant_Dic,All_seg_incident,Grid_center=Find_Best_Location(df_inrix,df_incident,All_seg_incident,width = 0.1,height = 0.1  )

#%%Run

for counter, [num_resources, alpha, model_i] in enumerate(HyperParams):
    print(counter, num_resources, alpha,model_i , model_i)
    
    df_responders_Exist=False
    Figure_Tag=False
    for row,row_values  in time_range.iloc[0:5].iteritems(): 
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
        Test=Dispaching_Scoring(DF_Test_metric_time.loc[row,:],df_responders,df_incident ,Window_size_hour,Delay,Penalty,model=model_i, Figure_Tag=False)
        DF_Test_metric_time.loc[row,Test.keys().tolist()]=Test.tolist()
        Figure_Tag=False




#%%
DF_Test_metric_time.mean()
DF_Test_metric_time.isna().sum()
Graph_Distance_Metric(DF_Test_metric_time,'Copared peformance of allocation model using different prediction methods')
DF_Test_metric_time.to_pickle('output/'+NAME+'+Distance_NoPenalty_'+model_i+'_alpha='+str(alpha)+'.pkl') 


#%% Just run the following code to complete the table for each alpha you want:
DF_Test_metric_time=pd.read_pickle('output/'+NAME+'+Distance_naive_'+model_i+'_alpha='+str(alpha)+'.pkl') 
DF_Test_metric_time.mean()

Table_Printer(pd.read_pickle('output/'+'DF_Test_metric_time_AllMethods_Naive_Alltestwindow'+'_Distance_Naive_NoPenalty_alpha='+str(0)+'.pkl') , 'naive')
Table_Printer(pd.read_pickle('output/'+'DF_Test_metric_time_AllMethods_Naive_Alltestwindow'+'_Distance_Naive_NoPenalty_alpha='+str(0.5)+'.pkl') , 'naive')
Table_Printer(pd.read_pickle('output/'+'DF_Test_metric_time_AllMethods_kmeans_Alltestwindow'+'+Distance_NoPenalty_naive_alpha='+str(2)+'.pkl') , 'naive')
Table_Printer(pd.read_pickle('output/'+'DF_Test_metric_time_AllMethods_kmeans_Alltestwindow'+'_Distance_Naive_NoPenalty_alpha='+str(0)+'.pkl') , 'Logistic_Regression+No_Resample')
Table_Printer(pd.read_pickle('output/'+'DF_Test_metric_time_AllMethods_kmeans_Alltestwindow'+'_Distance_Naive_NoPenalty_alpha='+str(1)+'.pkl') , 'Logistic_Regression+No_Resample')


def Table_Printer(DF_Test_metric_time, Name):

    MODEL0=Name+'_TotalNumberAccidents'
    MODEL1=Name+'_TotalNumberAccidentsNotResponded'
    MODEL2=Name+'_RespondedbyDistance'
    MODEL3=Name+'_RespondedbyDistanceperAccdient'
    
    '''
    MODEL0='naive_TotalNumberAccidents'
    MODEL1='naive_TotalNumberAccidentsNotResponded'
    MODEL2='naive_RespondedbyDistance'
    MODEL3='naive_RespondedbyDistanceperAccdient'
    '''
    
    
    DF_summarytable=pd.DataFrame(columns=['10V2h','10V1h','20V2h','20V1h'],
                                 index= ['Total Number of Accidents','Total Number of Not-responded Accidents',
                                         'Total Travel Distance (P=0)','Total Travel Distance Per Accident (P=0)',
                                         'Total Travel Distance (P=100)','Total Travel Distance Per Accident (P=100)',
                                         'Total Travel Distance (P=200)','Total Travel Distance Per Accident (P=200)'])
    DF_summarytable.loc['Total Number of Accidents']=DF_Test_metric_time[[MODEL0+'?10V2h',MODEL0+'?10V1h',MODEL0+'?20V2h',MODEL0+'?20V1h']].mean().tolist()
    DF_summarytable.loc['Total Number of Not-responded Accidents']=DF_Test_metric_time[[MODEL1+'?10V2h',MODEL1+'?10V1h',MODEL1+'?20V2h',MODEL1+'?20V1h']].mean().tolist()
    DF_summarytable.loc['Total Travel Distance (P=0)']=DF_Test_metric_time[[MODEL2+'?10V2h',MODEL2+'?10V1h',MODEL2+'?20V2h',MODEL2+'?20V1h']].mean().tolist()
    DF_summarytable.loc['Total Travel Distance Per Accident (P=0)']=DF_Test_metric_time[[MODEL2+'perAccdient?10V2h',MODEL2+'perAccdient?10V1h',MODEL2+'perAccdient?20V2h',MODEL2+'perAccdient?20V1h']].mean().tolist()
    
    
    Penalty=100
    DF_Test_metric_time[MODEL2+'?10V1h']=DF_Test_metric_time[MODEL2+'?10V1h']+Penalty*DF_Test_metric_time[MODEL1+'?10V1h']
    DF_Test_metric_time[MODEL2+'?10V2h']=DF_Test_metric_time[MODEL2+'?10V2h']+Penalty*DF_Test_metric_time[MODEL1+'?10V2h']
    DF_Test_metric_time[MODEL2+'?20V1h']=DF_Test_metric_time[MODEL2+'?20V1h']+Penalty*DF_Test_metric_time[MODEL1+'?20V1h']
    DF_Test_metric_time[MODEL2+'?20V2h']=DF_Test_metric_time[MODEL2+'?20V2h']+Penalty*DF_Test_metric_time[MODEL1+'?20V2h']
    
    DF_Test_metric_time[MODEL3+'?10V1h']=DF_Test_metric_time[MODEL2+'?10V1h']/DF_Test_metric_time[MODEL0+'?10V1h']
    DF_Test_metric_time[MODEL3+'?10V2h']=DF_Test_metric_time[MODEL2+'?10V2h']/DF_Test_metric_time[MODEL0+'?10V2h']
    DF_Test_metric_time[MODEL3+'?20V1h']=DF_Test_metric_time[MODEL2+'?20V1h']/DF_Test_metric_time[MODEL0+'?20V1h']
    DF_Test_metric_time[MODEL3+'?20V2h']=DF_Test_metric_time[MODEL2+'?20V2h']/DF_Test_metric_time[MODEL0+'?20V2h']
    DF_Test_metric_time.fillna(0,inplace=True)
    DF_Test_metric_time.mean()
    
    '''
    DF_Test_metric_time[[MODEL0+'?10V2h',MODEL0+'?10V1h',MODEL0+'?20V2h',MODEL0+'?20V1h']].mean()
    DF_Test_metric_time[[MODEL1+'?10V2h',MODEL1+'?10V1h',MODEL1+'?20V2h',MODEL1+'?20V1h']].mean()
    DF_Test_metric_time[[MODEL2+'?10V2h',MODEL2+'?10V1h',MODEL2+'?20V2h',MODEL2+'?20V1h']].mean()
    DF_Test_metric_time[[MODEL3+'?10V2h',MODEL3+'?10V1h',MODEL3+'?20V2h',MODEL3+'?20V1h']].mean()
    '''
    DF_summarytable.loc['Total Travel Distance (P=100)']=DF_Test_metric_time[[MODEL2+'?10V2h',MODEL2+'?10V1h',MODEL2+'?20V2h',MODEL2+'?20V1h']].mean().tolist()
    DF_summarytable.loc['Total Travel Distance Per Accident (P=100)']=DF_Test_metric_time[[MODEL3+'?10V2h',MODEL3+'?10V1h',MODEL3+'?20V2h',MODEL3+'?20V1h']].mean().tolist()
    
    
    
    
    Penalty=100 #another 100
    DF_Test_metric_time[MODEL2+'?10V1h']=DF_Test_metric_time[MODEL2+'?10V1h']+Penalty*DF_Test_metric_time[MODEL1+'?10V1h']
    DF_Test_metric_time[MODEL2+'?10V2h']=DF_Test_metric_time[MODEL2+'?10V2h']+Penalty*DF_Test_metric_time[MODEL1+'?10V2h']
    DF_Test_metric_time[MODEL2+'?20V1h']=DF_Test_metric_time[MODEL2+'?20V1h']+Penalty*DF_Test_metric_time[MODEL1+'?20V1h']
    DF_Test_metric_time[MODEL2+'?20V2h']=DF_Test_metric_time[MODEL2+'?20V2h']+Penalty*DF_Test_metric_time[MODEL1+'?20V2h']
    
    DF_Test_metric_time[MODEL3+'?10V1h']=DF_Test_metric_time[MODEL2+'?10V1h']/DF_Test_metric_time[MODEL0+'?10V1h']
    DF_Test_metric_time[MODEL3+'?10V2h']=DF_Test_metric_time[MODEL2+'?10V2h']/DF_Test_metric_time[MODEL0+'?10V2h']
    DF_Test_metric_time[MODEL3+'?20V1h']=DF_Test_metric_time[MODEL2+'?20V1h']/DF_Test_metric_time[MODEL0+'?20V1h']
    DF_Test_metric_time[MODEL3+'?20V2h']=DF_Test_metric_time[MODEL2+'?20V2h']/DF_Test_metric_time[MODEL0+'?20V2h']
    DF_Test_metric_time.fillna(0,inplace=True)
    DF_Test_metric_time.mean()
    
    DF_summarytable.loc['Total Travel Distance (P=200)']=DF_Test_metric_time[[MODEL2+'?10V2h',MODEL2+'?10V1h',MODEL2+'?20V2h',MODEL2+'?20V1h']].mean().tolist()
    DF_summarytable.loc['Total Travel Distance Per Accident (P=200)']=DF_Test_metric_time[[MODEL3+'?10V2h',MODEL3+'?10V1h',MODEL3+'?20V2h',MODEL3+'?20V1h']].mean().tolist()
    
    pd.options.display.float_format = "{:,.2f}".format
    print(DF_summarytable)
#%%





DF_Test_metric_time.to_pickle('output/'+NAME+'_Distance_Naive_NoPenalty.pkl') 
DF_Test_metric_time=pd.read_pickle('output/'+NAME+'_Distance_Naive_NoPenalty.pkl') 



DF_Test_metric_time[['Logistic_Regression+No_Resample_TotalNumberAccidentsNotResponded',MODEL1+'']].head(30)






