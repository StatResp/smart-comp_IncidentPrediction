# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 10:50:06 2021

@author: Sayyed Mohsen Vazirizade
This code reads the output results of allocation and distance evalution from different models, and put all of them in one DF.
Also it draws a barchart graph and genrate excel tables for the summary of the results. 
"""


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle5 as pickle

#num_resources_list =[10,15,20] #[2, 3]# [10,15,20]
#alpha_list=[0.0,0.5,1.0,2.0]  #[0,1]#[0.0,0.5,1.0,2.0]

num_resources_list =[10,15,20]
alpha_list=[0.0,0.5,1.0,2.0]


#%% Type 1
#model_list=['naive','LR+NoR+NoC1', 'LR+RUS+NoC1','LR+ROS+NoC1','LR+NoR+KM2','LR+RUS+KM2','LR+ROS+KM2']              #  'Logistic_Regression+No_Resample'#'Logistic_Regression+No_Resample'  #model_i='Naive'
#NAME='LR'
#%% Type 2
#model_list=['NN+NoR+NoC1', 'NN+RUS+NoC1','NN+ROS+NoC1','NN+NoR+KM2','NN+RUS+KM2','NN+ROS+KM2']  
#NAME='NN'
#%% Type 3
#model_list=['RF+CW+NoC1', 'RF+NoR+NoC1', 'RF+RUS+NoC1','RF+ROS+NoC1','RF+CW+KM2', 'RF+NoR+KM2', 'RF+RUS+KM2','RF+ROS+KM2']
#NAME='RF'
#%% Type 4
model_list=['Naive','LR+NoR+NoC1', 'LR+RUS+NoC1','LR+ROS+NoC1','LR+NoR+KM2','LR+RUS+KM2','LR+ROS+KM2',
            'NN+NoR+NoC1', 'NN+RUS+NoC1','NN+ROS+NoC1','NN+NoR+KM2','NN+RUS+KM2','NN+ROS+KM2',
            'RF+CW+NoC1', 'RF+NoR+NoC1', 'RF+RUS+NoC1','RF+ROS+NoC1','RF+CW+KM2', 'RF+NoR+KM2', 'RF+RUS+KM2','RF+ROS+KM2',
            'ZIP+NoR+NoC1','ZIP+RUS+NoC1','ZIP+ROS+NoC1','ZIP+NoR+KM2','ZIP+RUS+KM2','ZIP+ROS+KM2']

#
#model_list=['NN+NoR+NoC1', 'NN+RUS+NoC1','NN+ROS+NoC1','NN+NoR+KM2','NN+RUS+KM2','NN+ROS+KM2'] 
NAME='LR+NN+RF+ZIP'
#%% 
Delay=0.5 #1 #.5
Speed=100
Penalty=np.nan
Window_size_hour=4


HyperParams=[]
#DF_metric_all=pd.DataFrame(columns=['num_resources','alpha','model_i','DistanceTravel','TotalNumAccidents','TotalNumAccidentsNotResponded','DistanceTravelPerAccident'],index=range(len(num_resources_list)*len(alpha_list)*len(model_list)))

DF_metric_allmethod_time=pd.DataFrame(columns=['num_resources','alpha','model_i','DistanceTravel','TotalNumAccidents','TotalNumAccidentsNotResponded','DistanceTravelPerAccident'])

row_id=0
for num_resources in num_resources_list:
    for alpha in alpha_list:
        for model_i in model_list:
            print(row_id, num_resources,alpha,model_i)
            HyperParams.append([num_resources,alpha,model_i])
            DF_temp=pd.read_json( 'results/Distance/Distance-Metric_@'+model_i+'_'+str(num_resources)+'V'+str(Delay)+'h'+str(Speed)+'S'+str(alpha)+'alpha.json',orient='table')
            DF_temp['num_resources']=num_resources
            DF_temp['alpha']=alpha
            DF_temp['model_i']=model_i
            DF_temp['model_index']=model_list.index(model_i)
            #DF_metric_all.loc[row_id, ['DistanceTravel','TotalNumAccidents','TotalNumAccidentsNotResponded','DistanceTravelPerAccident']]=A
            DF_metric_allmethod_time=DF_metric_allmethod_time.append(DF_temp)
            row_id+=1

DF_metric_allmethod_time=DF_metric_allmethod_time.reset_index().drop('index', axis=1)

print(DF_metric_allmethod_time)
DF_metric_allmethod_time.to_pickle('results/Distance_'+NAME+'.pkl')
print('Done')
DF_metric_allmethod_time_Jan=pickle.load(open('results/Distance_LR+NN+RF+ZIP_Jan.pkl', 'rb')) 
DF_metric_allmethod_time_Dec=pickle.load(open('results/Distance_LR+NN+RF+ZIP_Dec.pkl', 'rb')) 
DF_metric_allmethod_time=DF_metric_allmethod_time_Jan.append(DF_metric_allmethod_time_Dec)
#%% 
#DF_metric_allmethod_time=pickle.load(open('results/Distance_LR+NN+RF_Jan.pkl', 'rb'))
#Building the table using the mean of all:
DF_metric_all_mean=DF_metric_allmethod_time.groupby(['model_i','alpha','num_resources','model_index']).mean().reset_index().sort_values(['model_index','num_resources','alpha']).reset_index().drop('index',axis=1)
DF_metric_all_max=DF_metric_allmethod_time.groupby(['model_i','alpha','num_resources','model_index']).max().reset_index().sort_values(['model_index','num_resources','alpha']).reset_index().drop('index',axis=1)

DF_Distance_DistanceTravel_mean=DF_metric_all_mean.pivot(index='model_i', columns=['num_resources','alpha'])['DistanceTravel'].loc[model_list].reset_index()

DF_NotResponded_TotalNumAccidentsNotResponded_mean=DF_metric_all_mean.pivot(index='model_i', columns=['num_resources','alpha'])['TotalNumAccidentsNotResponded'].loc[model_list].reset_index()
DF_NotResponded_TotalNumAccidentsNotResponded_max=DF_metric_all_max.pivot(index='model_i', columns=['num_resources','alpha'])['TotalNumAccidentsNotResponded'].loc[model_list].reset_index()

DF_DistancePerAccident_DistanceTravelPerAccident_mean=DF_metric_all_mean.pivot(index='model_i', columns=['num_resources','alpha'])['DistanceTravelPerAccident'].loc[model_list].reset_index()

with pd.ExcelWriter('results/AllcationResults_'+NAME+'.xlsx') as writer: 
    DF_Distance_DistanceTravel_mean.to_excel(writer, sheet_name='DistanceTravel_mean')
    DF_NotResponded_TotalNumAccidentsNotResponded_mean.to_excel(writer, sheet_name='TotalNumAccNotResp_mean')
    DF_NotResponded_TotalNumAccidentsNotResponded_max.to_excel(writer, sheet_name='TotalNumAccNotResp_max')
    DF_DistancePerAccident_DistanceTravelPerAccident_mean.to_excel(writer, sheet_name='DistanceTravelPerAcc_mean')

#%% 
#Building the bar chart:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
DF_metric_all
#sns.set_context('paper')




DF_metric_allmethod_time
def Box_plot(DF_metric_allmethod_time, y,num_resources ):
    DF_metric_allmethod_time['alpha ']='alpha = '+DF_metric_allmethod_time['alpha'].astype(str)
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    
    fig, ax = plt.subplots(figsize=(20,15))
    
    
#    DF_metric_allmethod_time['alpha ']='alpha='+DF_metric_allmethod_time['alpha'].astype(str)
#    fig=sns.barplot(x = 'model_i', y = 'DistanceTravel', hue='alpha ',  data = DF_metric_allmethod_time[DF_metric_allmethod_time['num_resources']==num_resources],
#                palette = 'hls', ci = 'sd' ,
#                capsize = .1, errwidth = 0.5,
#                ax= ax            )    
    
    fig=sns.boxplot(x = 'model_i', y = y, hue='alpha ',  data = DF_metric_allmethod_time[DF_metric_allmethod_time['num_resources']==num_resources],
                palette = 'hls',fliersize=1, linewidth=1,whis=100,
                ax= ax  )    
    
    plt.xticks(rotation=90)
    plt.legend(loc=9)
    #plt.yscale('log')
    ax.set_xlabel('Model')
    ax.set_ylabel('Average Travel Distance per Accident (km)')
    ax.set_title('p = '+str(num_resources)) 
    #ax.set_ylim(-220, 1300) 
    #ax.set_ylim(1, 5000)      

Box_plot(DF_metric_allmethod_time[DF_metric_allmethod_time['TotalNumAccidents']>0], 'DistanceTravelPerAccident',10);plt.savefig('DistanceTravelPerAccident_P=10.png')
Box_plot(DF_metric_allmethod_time[DF_metric_allmethod_time['TotalNumAccidents']>0], 'DistanceTravelPerAccident',15);plt.savefig('DistanceTravelPerAccident_P=15.png')
Box_plot(DF_metric_allmethod_time[DF_metric_allmethod_time['TotalNumAccidents']>0], 'DistanceTravelPerAccident',20);plt.savefig('DistanceTravelPerAccident_P=20.png')




#%% Analysis of Alpha
DF_metric_all=DF_metric_allmethod_time.copy()

DF_metric_all['model_i ']='models'
DF_metric_all.loc[DF_metric_all['model_i']=='Naive','model_i ']='Naive'
DF_metric_all_Added=DF_metric_all.groupby(['model_i ','alpha','num_resources']).mean().reset_index()
DF_metric_all_Added=DF_metric_all_Added[DF_metric_all_Added['model_i ']=='models']
DF_metric_all_Added['model_i']='mean of models'
DF_metric_all_Added['model_i ']='mean of models'
DF_metric_all=DF_metric_all.append(DF_metric_all_Added).reset_index().drop('index',axis=1)


palette={}
for i in DF_metric_all['model_i'].unique():
    palette[i]='grey'
palette['Naive']='red'
palette['mean of models']='blue'
sns.set_style('whitegrid')

DF_metric_all=DF_metric_all.rename(columns={'num_resources':'Number of the Resources'})
DF_metric_all=DF_metric_all.rename(columns={'DistanceTravel':'Travel Distance (km)'})
DF_metric_all=DF_metric_all.rename(columns={'DistanceTravelPerAccident':'Travel Distance per Accident (km)'})


sns.set(font_scale=2.5)
sns.set_style('whitegrid')
fig=sns.lmplot(x='alpha',y='Travel Distance (km)', hue='model_i', data = DF_metric_all,
           ci=None, order=2,scatter=False, line_kws={"lw":1.5}, col='Number of the Resources',sharey=False,
           palette=palette,
           legend=False,truncate=False,
           height=8, aspect=1
           )
#fig.fig.subplots_adjust(wspace=1)
plt.xlim(-0.25, 2.5)
axes = fig.axes
axes[0,0].set_ylim(450,600)   
axes[0,1].set_ylim(250,400)
axes[0,2].set_ylim(150,300)
#plt.ylim(450, 600)
#fig.set(ylim=(350, None))
plt.savefig('alpha.png')
