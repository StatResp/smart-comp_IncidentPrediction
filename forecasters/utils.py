# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:05:39 2021

@author: Sayyed Mohsen Vazirizade 
"""
import numpy as np
from pprint import pprint
from clustering.simple_clustering import Simple_Cluster
import pandas as pd
from plotting.figuring import Heatmap, Heatmap_on_Map_TimeRange
from reporting.generate_report import generate_report
import matplotlib.pyplot as plt
#%%
def update_meta_features(met_features, df_features, cat_col):
    """
    updates feature set with transformed names for categorical features and deletes old features + Intercept
    @param met_features: features from metadata
    @param df_features: features from the transformed dataframe
    @param cat_col: names of categorical columns
    @return:
    """
    try:
        for f in cat_col:
            f_cat = []
            for i in df_features:
                if "cat_" + f + '_' in i:
                    f_cat.append(i)
            met_features.remove(f)
            met_features.extend(f_cat)
    except:
        print('No Categorical Data Found!')
    if 'Intercept' not in met_features:    
        met_features.extend(['Intercept'])
    return met_features



#%%

def Results_Generator(model_type,model, Conf_Matrix, test_likelihood, test_likelihood_all, pred_likelihood, pred_likelihood_all,test_MSE, test_MSE_all, pred_MSE,pred_MSE_all):
    results = {   'test_likelihood': test_likelihood,
                  'test_likelihood_all': test_likelihood_all,
                  'predict_likelihood': pred_likelihood,
                  'predict_likelihood_all': pred_likelihood_all,
                  'test_MSE': test_MSE,
                  'test_MSE_all': test_MSE_all,
                  'predict_MSE': pred_MSE,     
                  'predict_MSE_all': pred_MSE_all, 
                  "accuracy": Conf_Matrix['accuracy'], "precision": Conf_Matrix['precision'], "recall": Conf_Matrix['recall'], "f1":Conf_Matrix['f1'],
                  "accuracy_all": Conf_Matrix['accuracy_all'], "precision_all": Conf_Matrix['precision_all'], "recall_all": Conf_Matrix['recall_all'], "f1_all":Conf_Matrix['f1_all'],
                  "threshold": Conf_Matrix['threshold'],
                  "threshold_all": Conf_Matrix['threshold_all']} 
    
    if (model_type in ['SR','LR','ZIP']):
        results['train_likelihood']= model.model_stats['train_likelihood']
        results['train_likelihood_all']= model.model_stats['train_likelihood_all']
        results['aic']= model.model_stats['aic']
        results['aic_all']= model.model_stats['aic_all']    
    else:
        results['train_likelihood']= None
        results['train_likelihood_all']= [None]
        results['aic']= None
        results['aic_all']= [None]
        
    
    
    
    return(results)




#%%


def Conf_Mat(df, metadata, current_model):
    '''
    This function calculates the confusion matrix for classification models

    Parameters
    ----------
    df : dataframe
        Includes all of our data.
    metadata : dict
        metadata.
    model_name : string
        name of the model.

    Returns
    -------
    Dic
        the calculated values for accuracy, precision, recall, and F1-score.

    '''
    '''
    Name_of_classification_Col='predicted_TF'
    df=df_test[[metadata['pred_name_Count'],metadata['pred_name_TF'],'predicted','predicted_TF','cluster_label','threshold']]

    df.sum()
    df.max()
    
    df[(df['predicted']>0.12186066301832887) & (df['cluster_label']==1)]
    sum(df_[metadata['pred_name_TF']]!=df_['predicted_TF'])
    learn_results['1']['Logistic_Regression+RUS']['model'].model_threshold
    df_test
    '''
    
    
    def A_P_R_F1(df, Name_of_classification_Col_Actual=metadata['pred_name_TF'], Name_of_classification_Col_Predicted='predicted_TF'):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score (df[Name_of_classification_Col_Actual], df[Name_of_classification_Col_Predicted]) # accuracy: (tp + tn) / (p + n)
        precision= precision_score(df[Name_of_classification_Col_Actual], df[Name_of_classification_Col_Predicted]) # precision tp / (tp + fp)
        recall   = recall_score   (df[Name_of_classification_Col_Actual], df[Name_of_classification_Col_Predicted]) # recall: tp / (tp + fn)
        f1       = f1_score       (df[Name_of_classification_Col_Actual], df[Name_of_classification_Col_Predicted]) # f1: 2 tp / (2 tp + fp + fn)        
        return accuracy, precision, recall, f1
    
    if ((current_model['model_type']=='LR') | (current_model['model_type']=='ZIP')):     #if the model can generate the columns predicted and predicted_TF, use this
        clusters=sorted(df.cluster_label.unique())
        accuracy, precision, recall, f1=A_P_R_F1(df )
        threshold=df['threshold'].mean()
        if len(clusters)==1:
            accuracy_all, precision_all, recall_all, f1_all, threshold_all=[np.nan], [np.nan], [np.nan], [np.nan],  [np.nan]
            
        else:
            accuracy_all, precision_all, recall_all, f1_all, threshold_all = [], [], [], [],[]
            for temp_cluster in clusters:
                accuracy_c, precision_c, recall_c, f1_c=A_P_R_F1(df[df['cluster_label']==temp_cluster])
                accuracy_all.append(accuracy_c)
                precision_all.append(precision_c)
                recall_all.append(recall_c)
                f1_all.append(f1_c)
                threshold_all.append(df[df['cluster_label']==temp_cluster]['threshold'].mean())
                
    elif current_model['model_type']=='Naive':   #non classification methods
        def Calculator_accuracy(row,model_i):
            if row[metadata['pred_name_TF']]>0:
                return 1*row[model_i]
            else:
                return 1-row[model_i]    
    
        accuracy= (df.apply(lambda row: Calculator_accuracy(row,'predicted'), axis=1)/len(df)).sum()
        precision=(df['predicted']*df[metadata['pred_name_TF']]/sum(df[metadata['pred_name_TF']])).sum()
        recall=   (df['predicted']*df[metadata['pred_name_TF']]/sum(df['predicted'])).sum()
        f1=2*recall*precision/(recall+precision)
        threshold=np.nan
        accuracy_all, precision_all, recall_all, f1_all, threshold_all=[np.nan], [np.nan], [np.nan], [np.nan],  [np.nan]
       
    
    else:   #non classification methods
        Length=len(df['cluster_label'].unique())
        accuracy, precision, recall, f1, threshold=np.nan, np.nan, np.nan, np.nan, np.nan
        accuracy_all, precision_all, recall_all, f1_all, threshold_all=[np.nan]*Length, [np.nan]*Length, [np.nan]*Length, [np.nan]*Length,  [np.nan]*Length

    
    
    return {"accuracy": accuracy, "accuracy_all": accuracy_all,
            "precision": precision, "precision_all": precision_all, 
            "recall": recall, "recall_all": recall_all,
            "f1":f1, "f1_all": f1_all,
            "threshold":threshold, "threshold_all": threshold_all}





#%%
def Naive_adder(DF_train,DF_test,learn_results,Window_Number_i ,metadata): 

    current_model={};current_model['model_type']='Naive'
    DF=DF_train[['XDSegID',metadata['pred_name_TF']]].groupby('XDSegID').agg({metadata['pred_name_TF']: ['count','mean']})
    DF.columns=['count', 'predicted']
    DF['predicted_TF']=None
    DF['cluster_label']=0
    DF_test=pd.merge(DF,DF_test[['XDSegID', 'time_local',metadata['pred_name_Count'],metadata['pred_name_TF']]], left_on= 'XDSegID'  ,right_on=  'XDSegID', how='right'  )
    Conf_Matrix = Conf_Mat(DF_test, metadata, current_model=current_model)  
    learn_results[Window_Number_i]['Naive']={}
    learn_results[Window_Number_i]['Naive']['df_test']=DF_test
    learn_results[Window_Number_i]['Naive']['results']=Results_Generator('Naive',None, Conf_Matrix, None, None, None, None,None, None, None,None)
    return learn_results


#%%
def Mean_of_AllTestTime_Results(learn_results):
    #Building a DF using the metrics and different test windows
    DF_results=pd.DataFrame()
    j=0
    for Window_Number_i in learn_results.keys():
        for model_i in learn_results[Window_Number_i].keys():
            print(model_i)
            for Parameter in learn_results[Window_Number_i][model_i].keys():
                if Parameter=='results':
                    #print(learn_results[Window_Number_i][model_i][Parameter])
                    DF_results.loc[j,'Window_Number']=Window_Number_i
                    DF_results.loc[j,'model']=model_i
                    for Metric in learn_results[Window_Number_i][model_i][Parameter].keys():
                        if isinstance(learn_results[Window_Number_i][model_i][Parameter][Metric], list):
                            LIST=[i for i in learn_results[Window_Number_i][model_i][Parameter][Metric]]
                            for i,_ in enumerate(LIST):
                                #print(i)
                                DF_results.loc[j,(Metric+'_'+str(i+1))] = LIST[i]
                        else:
                            #LIST=np.array(learn_results[Window_Number_i][model_i][Parameter][Metric])
                            DF_results.loc[j,Metric]=learn_results[Window_Number_i][model_i][Parameter][Metric]
                        #DF_results.loc[j,Metric]=LIST
                            

                    
                    j=j+1
    #Adding the mean values of the metrics of different test windows               
    for i in  DF_results['model'].unique():
            DF_results.loc[j,:]=DF_results[DF_results['model']==i].mean()
            DF_results.loc[j,'Window_Number']='Mean'
            DF_results.loc[j,'model']=i
            j=j+1  
    return  DF_results

       
def Concat_AllTestTime(learn_results,metadata):            
    DF=pd.DataFrame()
    #Briginging all of the different test windows in one dataframe
    for Window_Number_i in learn_results.keys():
        DF_Temporary=pd.DataFrame()
        for model_i  in learn_results[Window_Number_i].keys():
            DF_Temporary[model_i]=learn_results[Window_Number_i][model_i]['df_test']['predicted']
            DF_Temporary[model_i+'_TF']=learn_results[Window_Number_i][model_i]['df_test']['predicted_TF']
        DF_Temporary[['XDSegID','time_local',metadata['pred_name_Count'],metadata['pred_name_TF']]] =learn_results[Window_Number_i][model_i]['df_test'][['XDSegID','time_local',metadata['pred_name_Count'],metadata['pred_name_TF']]] 
        #DF_train=learn_results[Window_Number_i][model_i]['df_train'][['XDSegID',metadata['pred_name_TF']]].groupby('XDSegID').agg({metadata['pred_name_TF']: ['count','mean']})
        #DF_train.columns=['count', 'Naive']
        #DF_Temporary=pd.merge(DF_train,DF_Temporary, left_on= 'XDSegID'  ,right_on=  'XDSegID', how='right'  )
        DF_Temporary['Test_Group']=Window_Number_i
        DF=DF.append(DF_Temporary) 
    #DF['week']=DF['time_local'].dt.week        
                
    #Just reordering the columns
    
    Colunms=DF.columns.copy()  
    BegList=['XDSegID','time_local','Test_Group',metadata['pred_name_Count'],metadata['pred_name_TF']]
    BegList.extend(Colunms[Colunms.isin(BegList)==False].tolist())
    DF=DF[BegList]

       
    return  DF   




def Add_to_Dic(learn_results,DF_results):
    #Saving the mean values as a dictionary        
    learn_results['Mean']={}  
    for i in  DF_results['model'].unique():
        learn_results['Mean'][i]={} 
        learn_results['Mean'][i]['results']={}
        Columns=pd.Series(DF_results[(DF_results['Window_Number']=='Mean') & (DF_results['model']==i)].columns)
        Columns_search=list(learn_results['0'][i]['results'].keys())
        for j in Columns_search:
            if Columns.isin([j]).sum()==1:
                learn_results['Mean'][i]['results'][j]=DF_results[(DF_results['Window_Number']=='Mean') & (DF_results['model']==i)][j].iloc[0]
            else:
                search_str = '^' + j 
                learn_results['Mean'][i]['results'][j]=   (DF_results[(DF_results['Window_Number']=='Mean') & (DF_results['model']==i)][Columns[Columns.str.contains(search_str)].tolist()]).iloc[0].tolist()
    
      
    
    for Window_Number_i in  learn_results.keys():
       for model_i in  learn_results[Window_Number_i].keys(): 
           learn_results[Window_Number_i][model_i]['results']['spearman_corr']=DF_results[(DF_results['Window_Number']==Window_Number_i) & (DF_results['model']==model_i)]['spearman_corr'].iloc[0]
           learn_results[Window_Number_i][model_i]['results']['pearson_corr']=DF_results[(DF_results['Window_Number']==Window_Number_i) & (DF_results['model']==model_i)]['pearson_corr'].iloc[0]
        
    return learn_results





def Correlation_caluclator(DF, Columns_List,metadata,Type='spearman'):

    
    Time_aggregation=['Test_Group']
    #Time_aggregation=['week']
    Corr_Matrix=pd.DataFrame()   
    DF_margin_Space=DF[Time_aggregation+['XDSegID']+Columns_List].groupby(Time_aggregation+['XDSegID']).mean()
    for window_i in DF[Time_aggregation[0]].unique():             
        DF_Temporary=(DF_margin_Space.loc[(slice(window_i,window_i), slice(None)), :].corr(method = Type).loc[[metadata['pred_name_TF']]])
        DF_Temporary['Time_aggregation']=window_i
        DF_Temporary=DF_Temporary.drop(metadata['pred_name_TF'], axis=1)
        Corr_Matrix=Corr_Matrix.append(DF_Temporary)
      
        
    Corr_Matrix.loc['Mean'] = Corr_Matrix.mean()    
    Corr_Matrix.loc['Mean','Time_aggregation']='Mean'
    Corr_Matrix=Corr_Matrix.reset_index().drop('index',axis=1)

    return Corr_Matrix




def Correlation_Function(DF,DF_results,metadata):
    
    Columns_List=list(np.append(DF_results['model'].unique(),[metadata['pred_name_TF']]))
    #Columns_List=list(DF_results['model'].unique())
    Corr_Matrix_spearman=Correlation_caluclator(DF,Columns_List,metadata,Type='spearman')
    Corr_Matrix_pearson=Correlation_caluclator(DF,Columns_List,metadata,Type='pearson')
    
    DT1=pd.melt(Corr_Matrix_spearman, id_vars=['Time_aggregation'], value_vars=Columns_List[:-1])
    DT1=DT1.rename(columns={'value':'spearman_corr'})
    DT2=pd.melt(Corr_Matrix_pearson, id_vars=['Time_aggregation'], value_vars=Columns_List[:-1])
    DT2=DT2.rename(columns={'value':'pearson_corr'})
    DT=pd.merge(DT1,DT2,left_on=['Time_aggregation','variable'],right_on=['Time_aggregation','variable'], how='inner' )
    DT=DT.rename(columns={'variable':'model','Time_aggregation':'Window_Number'})
    DF_results=pd.merge(DF_results,DT,left_on=['Window_Number','model'],right_on=['Window_Number','model'], how='left' )
    
    
    #DF_results_[(DF_results_['Window_Number']==Window_Number_i) & (DF_results_['model']==model_i)]['spearman_corr'].iloc[0]

    return DF_results



   

    

def Metric_calculator_per_time(DF_Test_spacetime, DF_results,metadata):
    '''
    This function generates the figure that shows the accuracy, precision, reall, and F1-score using the naive model and the prediction models for 1 month

    Parameters
    ----------
    DF_Test_spacetime : DF
        DESCRIPTION.
    DF_results : DF
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
 
    from plotting.figuring import Graph_Metric
    def Calculator_accuracy(row,model_i):
        if row[metadata['pred_name_TF']]>0:
            return 1*row[model_i]
        else:
            return 1-row[model_i]
    
    '''
    DF_Temporary=pd.DataFrame()
    DF_Temporary['time_local']=DF_Test_spacetime['time_local']
    for time_local_i in DF_Test_spacetime['time_local'].drop_duplicates().sort_values():
        Mask=DF_Test_spacetime['time_local']==time_local_i
        for model_i  in (DF_results['model'].unique()): 
            DF_Temporary.loc[Mask,model_i+'_accuracy']=DF_Test_spacetime[Mask].apply(lambda row: Calculator_accuracy(row,model_i), axis=1)/len(DF_Test_spacetime[Mask])
            DF_Temporary.loc[Mask,model_i+'_recall']=DF_Test_spacetime[Mask][model_i]*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/((DF_Test_spacetime[Mask][metadata['pred_name_TF']]).sum())
            DF_Temporary.loc[Mask,model_i+'_precsion']=DF_Test_spacetime[Mask][model_i]*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/((DF_Test_spacetime[Mask][model_i+'_TF']).sum()) 
        DF_Temporary.loc[Mask,'Naive_accuracy']=DF_Test_spacetime[Mask].apply(lambda row: Calculator_accuracy(row,'Naive'), axis=1)/len(DF_Test_spacetime[Mask])
        DF_Temporary.loc[Mask,'Naive_recall']=DF_Test_spacetime[Mask]['Naive']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/sum(DF_Test_spacetime[Mask][metadata['pred_name_TF']])
        DF_Temporary.loc[Mask,'Naive_precsion']=DF_Test_spacetime[Mask]['Naive']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/sum(DF_Test_spacetime[Mask]['Naive'])
    DF_time=DF_Temporary.groupby('time_local').sum().reset_index()
    for model_i  in DF_results['model'].unique():
        DF_time[model_i+'_F1']=2*DF_time[model_i+'_recall']*DF_time[model_i+'_precsion']/(DF_time[model_i+'_recall']+DF_time[model_i+'_precsion'])
        DF_time[model_i+'_F1'].fillna(0, inplace = True) 
    DF_time['Naive_F1']=2*DF_time['Naive_recall']*DF_time['Naive_precsion']/(DF_time['Naive_recall']+DF_time['Naive_precsion'])
    DF_time['Naive_F1'].fillna(0, inplace = True) 
    DF_time.mean()    
    if metadata['figure_tag']==True:
        Graph_Metric(DF_time,'Comparing Total_Number_Incidents')
    '''
    
    
    
    DF_Temporary=pd.DataFrame()
    DF_Temporary['time_local']=DF_Test_spacetime['time_local']
    for time_local_i in DF_Test_spacetime['time_local'].drop_duplicates().sort_values():
        Mask=DF_Test_spacetime['time_local']==time_local_i
        for model_i  in (DF_results['model'].unique()): 
            if model_i!='Naive':
                DF_Temporary.loc[Mask,model_i+'_accuracy']=DF_Test_spacetime[Mask].apply(lambda row: Calculator_accuracy(row,model_i+'_TF'), axis=1)/len(DF_Test_spacetime[Mask])
                DF_Temporary.loc[Mask,model_i+'_recall']=DF_Test_spacetime[Mask][model_i+'_TF']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/((DF_Test_spacetime[Mask][metadata['pred_name_TF']]).sum())
                DF_Temporary.loc[Mask,model_i+'_precsion']=DF_Test_spacetime[Mask][model_i+'_TF']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/((DF_Test_spacetime[Mask][model_i+'_TF']).sum()) 
            elif model_i=='Naive':    
                DF_Temporary.loc[Mask,'Naive_accuracy']=DF_Test_spacetime[Mask].apply(lambda row: Calculator_accuracy(row,'Naive'), axis=1)/len(DF_Test_spacetime[Mask])
                DF_Temporary.loc[Mask,'Naive_recall']=DF_Test_spacetime[Mask]['Naive']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/sum(DF_Test_spacetime[Mask][metadata['pred_name_TF']])
                DF_Temporary.loc[Mask,'Naive_precsion']=DF_Test_spacetime[Mask]['Naive']*DF_Test_spacetime[Mask][metadata['pred_name_TF']]/sum(DF_Test_spacetime[Mask]['Naive'])
    DF_time_TF=DF_Temporary.groupby('time_local').sum().reset_index()
    for model_i  in DF_results['model'].unique():
        DF_time_TF[model_i+'_F1']=2*DF_time_TF[model_i+'_recall']*DF_time_TF[model_i+'_precsion']/(DF_time_TF[model_i+'_recall']+DF_time_TF[model_i+'_precsion'])
        DF_time_TF[model_i+'_F1'].fillna(0, inplace = True) 
    #DF_time_TF['Naive_F1']=2*DF_time_TF['Naive_recall']*DF_time_TF['Naive_precsion']/(DF_time_TF['Naive_recall']+DF_time_TF['Naive_precsion'])
    #DF_time_TF['Naive_F1'].fillna(0, inplace = True) 
    #DF_time_TF.mean()
    if metadata['figure_tag']==True:
        Graph_Metric(DF_time_TF,'Comparing Total_Number_Incidents_TF, Main Figure')
    

    
    return DF_time_TF


def Figure_Table_Generator(Window_Number_i,learn_results,metadata):
    #Drawing Figures and Generating Report
    #By default, the figures and report are generated for te last test time window. The ptions are:learn_results['0'].keys() and you an change it by change Window_Number_i=['0']
    results={}
    for m in range(len(metadata['model_type'])):
        model_i=metadata['model_type'][m]+'+'+metadata['resampling_type'][m]+'+'+metadata['cluster_type'][m]+str(metadata['cluster_number'][m])
        metadata['current_model']= {'Name':model_i, 'model_type': metadata['model_type'][m], 'resampling_type':metadata['resampling_type'][m], 'cluster_type': metadata['cluster_type'][m], 'cluster_number': metadata['cluster_number'][m]}
        df_test=learn_results[Window_Number_i][model_i]['df_test']
        #df_predict=learn_results[Window_Number_i][model_i]['df_predict']
        #time_range=df_test['time_local'].drop_duplicates().iloc[1:4].tolist()   #for drawing on the map, we have to specify a time frame
        time_range=df_test['time_local'].drop_duplicates().sort_values().tolist()  #for drawing on the map, we have to specify a time frame
        time_range
        metadata['start_time_test']   
        #Heatmap(        learn_results[model_i]['df_train'],metadata)
        Heatmap(        df_test,metadata,model_i+'_'+'testwindow('+Window_Number_i+')_'+': Actual Data', COLNAME=metadata['pred_name_TF'])
        plt.savefig('output/spatial_temporal_'+model_i+'_'+'testwindow('+Window_Number_i+')_'+'Actual Data.png')
        Heatmap(        df_test,metadata,model_i+'_'+'testwindow('+Window_Number_i+')_'+': Prediction',  COLNAME='predicted')   
        plt.savefig('output/spatial_temporal_'+model_i+'_'+'testwindow('+Window_Number_i+')_'+'Prediction.png')
        if metadata['current_model']['model_type'] in metadata['Count_Models']:
            Heatmap(        df_test,metadata,model_i+'_'+'testwindow('+Window_Number_i+')_'+'Actual Data', COLNAME=metadata['pred_name_Count'],maxrange=max(1, df_test[[metadata['pred_name_Count'],'predicted_Count']].max().max()))
            plt.savefig('output/spatial_temporal_'+model_i+'_'+'testwindow('+Window_Number_i+')_'+': Actual Data.png')
            Heatmap(        df_test,metadata,model_i+'_'+'testwindow('+Window_Number_i+')_'+'Prediction',  COLNAME='predicted_Count',maxrange=max(1, df_test[[metadata['pred_name_Count'],'predicted_Count']].max().max()))  
            plt.savefig('output/spatial_temporal_'+model_i+'_'+'testwindow('+Window_Number_i+')_'+': Prediction.png')
        Map=Heatmap_on_Map_TimeRange(df_test,time_range,metadata,Feature_List=[metadata['pred_name_TF']]+['predicted','cluster_label'],Name=model_i) ;  
        Map.save('output/Map_rate_'+model_i+'_'+'testwindow('+Window_Number_i+')_'+time_range[0].strftime('%Y-%m-%d %H')+'.html')  
        #print(learn_results[model_i]['model'].model_params[0].summary())               
        plt.show() 
        results[model_i]=learn_results[Window_Number_i][model_i]['results']   
    generate_report(results,metadata['cluster_number']+[1],'output/Report_'+metadata['model_type'][0]+'_'+'_testwindow('+Window_Number_i+')'+'.html' )   


#def predict(model, df, metadata):
#    #smv: not needed anymore
#    """
#    Wrapper method before data is passed to specific predict methods for regression models
#    @param model: the trained model
#    @param df: dataframe of points where predictions need to be made
#    @param metadata: dictionary with start and end dates for predictions, cluster labels etc
#    @return: dataframe with E(y|x) appended to each row
#    """
#    if model.name == 'Poisson_Regression' or model.name == 'Negative_Binomial_Regression' or model.name == 'Simple_Regression':
#        #df_ = create_regression_df_test(df, metadata)
#        df_=df
#        df_samples = model.predict(df_, metadata)
#        return df_samples