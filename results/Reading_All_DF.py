# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 09:32:17 2021

@author: Sayyed Mohsen Vazirizade




Since different machines and different versions of the code were used to generate the results for LR, NN, RF, and ZIP models; there might be some discrepancies in joining the results.
This code will handle this part for us and brings all the results together. 
This function reads all the DF_Test_spacetime from various prediction models and put them in 1 DF
The final generated DF, DF_Test_spacetime_All, will be used by allocation model. 
"""
import pickle5 as pickle
import pandas as pd
#%%
#LR
model_list_LR= ['naive','LR+NoR+NoC1', 'LR+RUS+NoC1','LR+ROS+NoC1','LR+NoR+KM2','LR+RUS+KM2','LR+ROS+KM2']   
#ZIP
model_list_ZIP= ['ZIP+NoR+NoC1', 'ZIP+RUS+NoC1','ZIP+ROS+NoC1','ZIP+NoR+KM2','ZIP+RUS+KM2','ZIP+ROS+KM2']   
#RF
model_list_RF= ['RF+CW+NoC1', 'RF+NoR+NoC1', 'RF+RUS+NoC1','RF+ROS+NoC1','RF+CW+KM2', 'RF+NoR+KM2', 'RF+RUS+KM2','RF+ROS+KM2']
#NN
model_list_NN= ['NN+NoR+NoC1', 'NN+RUS+NoC1','NN+ROS+NoC1','NN+NoR+KM2','NN+RUS+KM2','NN+ROS+KM2']   



File='results/12_Month/'
#%%
def Reader(model,Resample, Cluster ):
    #Reading the files
    Address=File+model+'/'+Cluster+'/'+Resample+'/'
    print(Address)
    try:
        #DF=pd.read_pickle(Address+'DF_Test_spacetime_AllMethods_kmeans_Alltestwindow.pkl')
        DF_Test_spacetime=pickle.load(open(Address+'DF_Test_spacetime_AllMethods_kmeans_Alltestwindow.pkl', 'rb')) 
        DF_results=pickle.load(open(Address+'DF_results_AllMethods_kmeans_Alltestwindow.pkl', 'rb')) 
    except:
        #DF=pd.read_pickle(Address+'DF_Test_spacetime_AllMethods_None_Alltestwindow.pkl')
        DF_Test_spacetime=pickle.load(open(Address+'DF_Test_spacetime_AllMethods_None_Alltestwindow.pkl', 'rb')) 
        DF_results=pickle.load(open(Address+'DF_results_AllMethods_None_Alltestwindow.pkl', 'rb')) 
    return DF_Test_spacetime,DF_results

#%% RF
def Column_Name_Fixer(DF,COL_Name,Type='rf'):
    DF=DF.rename(columns={Type+'+No_Resample':COL_Name,Type+'+No_Resample_TF':COL_Name+'_TF'})
    DF=DF.rename(columns={Type+'+RUS':COL_Name,Type+'+RUS_TF':COL_Name+'_TF'})
    DF=DF.rename(columns={Type+'+ROS':COL_Name,Type+'+ROS_TF':COL_Name+'_TF'})  
    return DF

def One_Column_value_Fixer(DF,COL_Name,Type='rf'):
    DF['model']=DF['model'].mask(DF['model']==Type+'+No_Resample', COL_Name)
    DF['model']=DF['model'].mask(DF['model']==Type+'+RUS', COL_Name)
    DF['model']=DF['model'].mask(DF['model']==Type+'+ROS', COL_Name)
    
    
    DF=DF.rename(columns={Type+'+No_Resample':COL_Name,Type+'+No_Resample_TF':COL_Name+'_TF'})
    DF=DF.rename(columns={Type+'+RUS':COL_Name,Type+'+RUS_TF':COL_Name+'_TF'})
    DF=DF.rename(columns={Type+'+ROS':COL_Name,Type+'+ROS_TF':COL_Name+'_TF'})  
    return DF

    
def Add_New_Data_rf(ID,DF_Test_spacetime_RF,DF_results_RF,model_list,Type='rf'):
    #fixing the column name and saving
    model=model_list[ID].split('+')[0];Resample=model_list[ID].split('+')[1];Cluster=model_list[ID].split('+')[2] 
    print(ID, model_list[ID],model,Resample, Cluster )
    DF_Test_spacetime,DF_results =  Reader(model,Resample, Cluster )
    COL_Name=model+'+'+Resample+'+'+Cluster
    print(DF_Test_spacetime.columns[3:5])
    DF_Test_spacetime=Column_Name_Fixer(DF_Test_spacetime,COL_Name,Type=Type)
    DF_results=One_Column_value_Fixer(DF_results,COL_Name,Type=Type)
    if len(DF_Test_spacetime_RF)>0:
        DF_Test_spacetime_RF=pd.merge(DF_Test_spacetime_RF, DF_Test_spacetime, left_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'count', 'naive'], right_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'count', 'naive'], how='left')
    else:
        DF_Test_spacetime_RF=DF_Test_spacetime[['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'count', 'naive',COL_Name,COL_Name+'_TF']]
    
    DF_results_RF=DF_results_RF.append(DF_results)
    DF_results_RF=DF_results_RF.reset_index().drop('index',axis=1)
    return DF_Test_spacetime_RF,DF_results_RF



DF_Test_spacetime_RF=pd.DataFrame()
DF_results_RF=pd.DataFrame()
for ID in range(0,8):
    DF_Test_spacetime_RF,DF_results_RF=Add_New_Data_rf(ID,DF_Test_spacetime_RF,DF_results_RF,model_list_RF,Type='rf')
    print('\n')   



DF_Test_spacetime_RF=DF_Test_spacetime_RF.drop('count',axis=1).rename(columns={'naive':'Naive'})
DF_Test_spacetime_RF.to_pickle(File+'RF/DF_Test_spacetime_RF'+'.pkl') 
DF_results_RF.to_pickle(File+'RF/DF_results_RF'+'.pkl') 
#%% NN
#DF_Test_spacetime_NN1=pd.read_pickle(File+'NN/DF_Test_spacetime_AllMethods_kmeans_Alltestwindow.pkl')
DF_Test_spacetime_NN1=pickle.load(open(File+'NN/KM2/DF_Test_spacetime_AllMethods_kmeans_Alltestwindow.pkl', 'rb')) 
DF_results_NN1=pickle.load(open(File+'NN/KM2/DF_results_AllMethods_kmeans_Alltestwindow.pkl', 'rb')) 
DF_Test_spacetime_NN1=DF_Test_spacetime_NN1.rename(columns={'NN+No_Resample':'NN+NoR+KM2',
                                                              'NN+RUS':'NN+RUS+KM2',
                                                              'NN+ROS':'NN+ROS+KM2',
                                                              'NN+No_Resample_TF':'NN+NoR+KM2_TF',
                                                              'NN+RUS_TF':'NN+RUS+KM2_TF',
                                                              'NN+ROS_TF':'NN+ROS+KM2_TF'})
DF_results_NN1['model']=DF_results_NN1['model'].mask(DF_results_NN1['model']=='NN+No_Resample', 'NN+NoR+KM2')    
DF_results_NN1['model']=DF_results_NN1['model'].mask(DF_results_NN1['model']=='NN+RUS', 'NN+RUS+KM2')    
DF_results_NN1['model']=DF_results_NN1['model'].mask(DF_results_NN1['model']=='NN+ROS', 'NN+ROS+KM2')      
    
    

#DF_Test_spacetime_NN2=pd.read_pickle(File+'NN/DF_Test_spacetime_AllMethods_None_Alltestwindow.pkl')
DF_Test_spacetime_NN2=pickle.load(open(File+'NN/NoC1/DF_Test_spacetime_AllMethods_None_Alltestwindow.pkl', 'rb')) 
DF_results_NN2=pickle.load(open(File+'NN/NoC1/DF_results_AllMethods_None_Alltestwindow.pkl', 'rb')) 
DF_Test_spacetime_NN2=DF_Test_spacetime_NN2.rename(columns={'NN+No_Resample':'NN+NoR+NoC1',
                                                              'NN+RUS':'NN+RUS+NoC1',
                                                              'NN+ROS':'NN+ROS+NoC1',
                                                              'NN+No_Resample_TF':'NN+NoR+NoC1_TF',
                                                              'NN+RUS_TF':'NN+RUS+NoC1_TF',
                                                              'NN+ROS_TF':'NN+ROS+NoC1_TF'})
DF_results_NN2['model']=DF_results_NN2['model'].mask(DF_results_NN2['model']=='NN+No_Resample', 'NN+NoR+NoC1')    
DF_results_NN2['model']=DF_results_NN2['model'].mask(DF_results_NN2['model']=='NN+RUS', 'NN+RUS+NoC1')    
DF_results_NN2['model']=DF_results_NN2['model'].mask(DF_results_NN2['model']=='NN+ROS', 'NN+ROS+NoC1') 
    
    
    
DF_Test_spacetime_NN=pd.merge(DF_Test_spacetime_NN1,DF_Test_spacetime_NN2,left_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'count', 'naive'], right_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'count', 'naive'], how='left' )
DF_Test_spacetime_NN=DF_Test_spacetime_NN.drop('count',axis=1).rename(columns={'naive':'Naive'})
DF_Test_spacetime_NN.to_pickle(File+'NN/DF_Test_spacetime_NN'+'.pkl') 

DF_results_NN=DF_results_NN1.append(DF_results_NN2).reset_index().drop('index',axis=1)
DF_results_NN.to_pickle(File+'NN/DF_results_NN'+'.pkl') 
#%% LR
DF_Test_spacetime_LR1=pickle.load(open(File+'LR/KM2/DF_Test_spacetimeLR.pkl', 'rb')) 
DF_Test_spacetime_LR2=pickle.load(open(File+'LR/NoC1/DF_Test_spacetimeLR.pkl', 'rb')) 

DF_results_LR1=pickle.load(open(File+'LR/KM2/DF_resultsLR.pkl', 'rb')) 
DF_results_LR2=pickle.load(open(File+'LR/NoC1/DF_resultsLR.pkl', 'rb')) 
DF_results_LR2=DF_results_LR2[DF_results_LR2['model']!='Naive']

DF_Test_spacetime_LR=pd.merge(DF_Test_spacetime_LR1,DF_Test_spacetime_LR2,left_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'Naive', 'Naive_TF'], right_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'Naive', 'Naive_TF'], how='left' )
DF_Test_spacetime_LR=DF_Test_spacetime_LR.drop('Naive_TF',axis=1)
DF_results_LR=DF_results_LR1.append(DF_results_LR2).reset_index().drop('index',axis=1)

DF_Test_spacetime_LR.to_pickle(File+'LR/DF_Test_spacetime_LR'+'.pkl') 
DF_results_LR.to_pickle(File+'LR/DF_results_LR'+'.pkl') 
#%% ZIP
DF_Test_spacetime_ZIP1=pickle.load(open(File+'ZIP/KM2/DF_Test_spacetimeZIP.pkl', 'rb')) 
DF_Test_spacetime_ZIP2=pickle.load(open(File+'ZIP/NoC1/DF_Test_spacetimeZIP.pkl', 'rb')) 

DF_results_ZIP1=pickle.load(open(File+'ZIP/KM2/DF_resultsZIP.pkl', 'rb')) 
DF_results_ZIP2=pickle.load(open(File+'ZIP/NoC1/DF_resultsZIP.pkl', 'rb')) 

DF_Test_spacetime_ZIP=pd.merge(DF_Test_spacetime_ZIP1,DF_Test_spacetime_ZIP2,left_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF'], right_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF'], how='left' )
DF_results_ZIP=DF_results_ZIP1.append(DF_results_ZIP2).reset_index().drop('index',axis=1)

DF_Test_spacetime_ZIP.to_pickle(File+'ZIP/DF_Test_spacetime_ZIP'+'.pkl') 
DF_results_ZIP.to_pickle(File+'ZIP/DF_results_ZIP'+'.pkl') 
#%% Combining 
'''
#DF_Test_spacetime_LR=pd.read_pickle(File+'LR/DF_Test_spacetime_LR.pkl') 
DF_Test_spacetime_LR=pickle.load(open(File+'NN/DF_Test_spacetime_LR.pkl', 'rb')) 
#DF_Test_spacetime_ZIP=pd.read_pickle(File+'LR/DF_Test_spacetime_ZIP.pkl') 
DF_Test_spacetime_ZIP=pickle.load(open(File+'NN/DF_Test_spacetime_ZIP.pkl', 'rb')) 
#DF_Test_spacetime_RF=pd.read_pickle(File+'RF/DF_Test_spacetime_RF.pkl') 
DF_Test_spacetime_RF=pickle.load(open(File+'NN/DF_Test_spacetime_RF.pkl', 'rb')) 
#DF_Test_spacetime_NN=pd.read_pickle(File+'NN/DF_Test_spacetime_NN.pkl') 
DF_Test_spacetime_NN=pickle.load(open(File+'NN/DF_Test_spacetime_NN.pkl', 'rb')) 
'''
#DF_Test_spacetime=DF_Test_spacetime_LR
DF_Test_spacetime=pd.merge(DF_Test_spacetime_LR, DF_Test_spacetime_ZIP, left_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF'], right_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF'], how='left')
DF_Test_spacetime=pd.merge(DF_Test_spacetime   , DF_Test_spacetime_RF , left_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'Naive'], right_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'Naive'], how='left')
DF_Test_spacetime=pd.merge(DF_Test_spacetime   , DF_Test_spacetime_NN , left_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'Naive'], right_on=['XDSegID', 'time_local', 'Test_Group', 'Total_Number_Incidents','Total_Number_Incidents_TF', 'Naive'], how='left')
DF_Test_spacetime.to_pickle(File+'DF_Test_spacetime_All.pkl') 


Coulmns_name=['model','Window_Number','accuracy','precision','recall','f1','pearson_corr','spearman_corr']
#DF_results=DF_results_LR[Coulmns_name]
DF_results=DF_results_LR[Coulmns_name].append(DF_results_ZIP[Coulmns_name])
DF_results=DF_results[Coulmns_name].append(DF_results_RF[Coulmns_name])
DF_results=DF_results[Coulmns_name].append(DF_results_NN[Coulmns_name]).reset_index().drop('index',axis=1)
DF_results.to_pickle(File+'DF_results_All.pkl') 
DF_Test_spacetime=pickle.load(open(File+'DF_Test_spacetime_All.pkl', 'rb')) 
#%%
#__________________________________________________________________________
#Just Saving for Decemeber
DF=DF_Test_spacetime[(DF_Test_spacetime['time_local']>=pd.Timestamp(year=2019, month=12, day=1, hour=0)) & (DF_Test_spacetime['time_local']<pd.Timestamp(year=2020, month=1, day=1, hour=0))]
DF.to_pickle(File+'Dec/DF_Test_spacetime_All_Dec.pkl') 
#Just Saving for January
DF=DF_Test_spacetime[(DF_Test_spacetime['time_local']>=pd.Timestamp(year=2020, month=1, day=1, hour=0))]
DF.to_pickle(File+'Jan/DF_Test_spacetime_All_Jan.pkl') 
#__________________________________________________________________________
order=['Naive',
       'LR+NoR+NoC1','LR+RUS+NoC1','LR+ROS+NoC1',
       'LR+NoR+KM2','LR+RUS+KM2','LR+ROS+KM2',   
       'NN+NoR+NoC1','NN+RUS+NoC1','NN+ROS+NoC1',
       'NN+NoR+KM2','NN+RUS+KM2','NN+ROS+KM2',            
       'RF+NoR+NoC1','RF+RUS+NoC1','RF+ROS+NoC1', 'RF+CW+NoC1',
       'RF+NoR+KM2','RF+RUS+KM2','RF+ROS+KM2', 'RF+CW+NoC1',            
       'ZIP+NoR+NoC1','ZIP+RUS+NoC1','ZIP+ROS+NoC1',
       'ZIP+NoR+KM2','ZIP+RUS+KM2','ZIP+ROS+KM2']   
DF_results_mean=(DF_results[DF_results['Window_Number']=='Mean'])
print(DF_results_mean.set_index('model').loc[order])
with pd.ExcelWriter(File+'Perfromance.xlsx') as writer: 
    DF_results_mean.set_index('model').loc[order].to_excel(writer, sheet_name='Perfromance(table2)')
#%%
DF_results['model_type']=DF_results.apply(lambda row: row['model'].split('+')[0], axis=1)


def Box_plot(DF_results, y, Tag ):
    sns.set_style('whitegrid')
    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(20,10))
    
    if Tag=='boxplot':
        fig=sns.boxplot(x = 'model', y = 'accuracy',hue='model_type', order=order, data = DF_results[DF_results['Window_Number']!='Mean'],
                    width=1,
                    palette = 'tab10',fliersize=1, linewidth=1,whis=100,
                    ax= ax)         
        
    elif Tag=='barplot':   
        fig=sns.barplot(x = 'model', y = 'accuracy', hue='model_type', order=order,  data = DF_results[DF_results['Window_Number']!='Mean'],
                    palette = 'tab10', ci = 'sd' ,
                    capsize = .5, errwidth = 0.5,
                    ax= ax            )
    ax.legend_.remove()
    plt.xticks(rotation=90)              
    ax.set_ylim(0.91, 0.97) 

    ax.set_xlabel('Model')
    ax.set_ylabel(y.capitalize())
    #ax.set_title('')    
    
Box_plot(DF_results,'accuracy' , 'barplot')
#%%



