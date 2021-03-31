"""
@Author - Sayyed Mohsen Vazirizade and Ayan Mukhopadhyay
Sample file that substitutes requests from the front end
"""

#clear the screen______________________________________________________________
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()



from pprint import pprint
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import _pickle as cPickle
import bz2
#from clustering.hierarchical_clustering import Hierarchical_Cluster
#from clustering.learning_based_hierarchical import Hierarchical_Cluster_L
#from clustering.simple_clustering import Simple_Cluster
#from clustering.outlier_detection import detect_outliers
from forecasters.regression_models import learn
from forecasters.utils import  Conf_Mat, Concat_AllTestTime, Mean_of_AllTestTime_Results, Correlation_Function, Add_to_Dic, Metric_calculator_per_time, Results_Generator,Naive_adder, Figure_Table_Generator
from allocation.Allocation import Dispaching_Scoring
from readConfig import read_config
import os
from copy import deepcopy
import matplotlib.pyplot as plt


#should be fixed:
   #1) Metric_calculator_per_time
   #2) Conf_Mat
   #3) y_verif_hat=model.predict(temp_cluster[~mask][metadata['features_ALL']])
print('mohsen started')
print('getcwd:      ', os.getcwd())
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)


#cwd ="C:\\Users\\smvaz\\Desktop\\synch folder\\Vanderbilt\\Geof\\prediction new 2" #Get current directory
getcwd=os.getcwd()
os.chdir(getcwd)
print('getcwd:      ', os.getcwd())
import random
random.seed(0)

def compressed_pickle(title, data):
 with bz2.BZ2File(title + '.pbz2', 'w') as f: 
 	cPickle.dump(data, f)

# Load any compressed pickle file
def decompress_pickle(file):
 data = bz2.BZ2File(file, 'rb')
 data = cPickle.load(data)
 return data



def MetaDataFixer(metadata):
    metadata['TF_Models']=['LR','RF','NN','SVM']
    metadata['Count_Models']=['ZIP','SR','PR']
    metadata['GLM_Models']=['SR','LR','PR', 'NBR','ZIP']
    return metadata

def prepare_sample_data_TDOT_BigDF(metadata):
    #for reading the incident data frame from a pickle file 
    print('read_DF')
    df_ = pickle.load(open(metadata['regressiondf_pickle_address'], 'rb'))
    #Histroically we have 2 types of DF, here we fix the name of the columns for time:
    if 'start_time' in (df_.columns.tolist()):
        df_=df_.rename(columns={'start_time':'time', 'time__':'time_local', 'Congestion': 'congestion_mean'})
        print('some column names have been changed: time__ -->  time_local   &   start_time -->  time    &   Congestion --> congestion_mean')
        df_=df_.rename(columns={'count':'Total_Number_Incidents_TF'})
        
        
    if (metadata['time_zone']).lower()=='local':
        metadata['time_column']='time_local'
    elif (metadata['time_zone']).lower()=='utc':
        metadata['time_column']='time'
    df_=df_.drop('time',axis=1) 
    #df_=df_.iloc[0:100000]
    #df_=df_[df_['Total_Number_Incidents']>0].reset_index().drop('index',axis=1)
    '''
    df_ = df_.sort_values(by=['time'])
    df_
    #just to make it faster
    TIMES=min(metadata['start_time_train'], metadata['start_time_test']) # TIMES=pd.Timestamp(year=2019, month=9, day=1, hour=0, tz='UTC')
    TIMEE=max(metadata['end_time_train'], metadata['end_time_test'])  # TIMEE=pd.Timestamp(year=2019, month=9, day=1, hour=0, tz='UTC')
    Orderbeg=df_[metadata['time_column']].searchsorted(TIMES,side='left')
    Orderend=df_[metadata['time_column']].searchsorted(TIMEE,side='right')
    df_=df_.iloc[Orderbeg:Orderend+1]    #df_ = df_.iloc[500:1500]    #for making the code faster
    '''
    #df_['cluster_label'] = 1              #we should change this one later
    metadata['units'] = np.sort(df_[metadata['unit_name']].unique()) #finding the unique metadata['unit_name'], which means unique locations
    metadata['location_map'] = {x: 1 for x in metadata['units']} 
    return df_, metadata



def Moving_Function(metadata):
    Window_Test=[]
    if metadata['train_test_type']=='moving_window':
        str(metadata['start_time_test']+pd.DateOffset(months=1))
        Delta=(metadata['end_time_test'].year-    metadata['start_time_test'].year)*12  +  (metadata['end_time_test'].month -    metadata['start_time_test'].month)
        for i in range(Delta+1):
            #Window_Train.append((metadata['end_time_train']+pd.DateOffset(months=i)))
            Window_Test.append((metadata['start_time_test']+pd.DateOffset(months=i)))
        
    else:
        Window_Test=[metadata['start_time_test'], metadata['end_time_test']]

    return Window_Test
    
#%%

if __name__ == "__main__":
        print('\nbeggining:')
        #model_name can be ['Simple_Regression','Poisson_Regression', 'Negative_Binomial_Regression', 'Zero_Inflated_Poisson_Regression'] ['Survival_Regression']
        metadata = read_config("config/params.conf")
        
        #df_, meta = prepare_sample_data_TDOT_auto(metadata)
        df_, meta = prepare_sample_data_TDOT_BigDF(metadata)
        #just to save time, the regressiondf can be saved in advance and be used here
        
        #try: os.remove(metadata['regressiondf_pickle_address']); print("file regressiondf Removed!");
        #except: print("Nothing Found!")
        results = {}
        learn_results={}
        Window_Test=Moving_Function(metadata)
        metadata=MetaDataFixer(metadata)
        
        for Window_Number_counter in range(len(Window_Test)-1):
                Window_Number_i=str(Window_Number_counter)
                if metadata['train_test_type']=='moving_window':
                    #metadata['start_time_train']
                    metadata['end_time_train']=Window_Test[Window_Number_counter]
                    metadata['start_time_test']=Window_Test[Window_Number_counter]
                    metadata['end_time_test']=Window_Test[Window_Number_counter+1]
                    
                
                
                
                learn_results[Window_Number_i]={}
                for m in range(len(metadata['model_type'])):
                    metadata['features_ALL'] = deepcopy(metadata['features'])        #feature_ALL incorporates Intercept,Categorical features, and orginal feature
                    model_i=metadata['model_type'][m]+'+'+metadata['resampling_type'][m]+'+'+metadata['cluster_type'][m]+str(metadata['cluster_number'][m])
                    metadata['current_model']= {'Name':model_i, 'model_type': metadata['model_type'][m], 'resampling_type':metadata['resampling_type'][m], 'cluster_type': metadata['cluster_type'][m], 'cluster_number': metadata['cluster_number'][m]}
                    metadata['current_model']['Name']
                    print('---------------------------------------------------------------------------------------')
                    print('\n',Window_Number_i, model_i, metadata['current_model']['Name'])
                    learn_results[Window_Number_i][model_i] = learn(df_, meta, current_model=metadata['current_model'])
                    model = learn_results[Window_Number_i][model_i]['model']
                    #df_test=learn_results[Window_Number_i][model_i]['df_test']
                    #df_pred=learn_results[Window_Number_i][model_i]['df_predict']
                    #df_verif=learn_results[Window_Number_i][model_i]['df_verif']
                    learn_results[Window_Number_i][model_i]['df_test'], test_likelihood_all,test_likelihood,test_MSE_all,test_MSE = model.prediction(df=learn_results[Window_Number_i][model_i]['df_test'], metadata=metadata)
                    learn_results[Window_Number_i][model_i]['df_predict'], pred_likelihood_all,pred_likelihood,pred_MSE_all,pred_MSE = model.prediction(df=learn_results[Window_Number_i][model_i]['df_predict'], metadata=metadata)
                    Conf_Matrix = Conf_Mat(learn_results[Window_Number_i][model_i]['df_test'], meta, current_model=metadata['current_model'])
                    #Conf_Matrix = Conf_Mat(df_pred, meta, model_name=metadata['models'][m])

                    learn_results[Window_Number_i][model_i]['results']=Results_Generator(metadata['current_model']['model_type'], model, Conf_Matrix, test_likelihood, test_likelihood_all, pred_likelihood, pred_likelihood_all,test_MSE, test_MSE_all, pred_MSE,pred_MSE_all)
                #adding the naive model in each test window
                learn_results=Naive_adder(learn_results[Window_Number_i][model_i]['df_train'],learn_results[Window_Number_i][model_i]['df_test'],learn_results,Window_Number_i ,metadata)
                
        #END OF FOR LOOP______________________________________________________
        #learn_results_backup=learn_results.copy()
        DF_Test_spacetime=Concat_AllTestTime(learn_results,metadata)
        DF_results =Mean_of_AllTestTime_Results(learn_results)
        DF_results=Correlation_Function(DF_Test_spacetime,DF_results,metadata)
        DF_Test_metric_time=Metric_calculator_per_time(DF_Test_spacetime,DF_results,metadata) #it also adds the metrics for naive model
        learn_results=Add_to_Dic(learn_results,DF_results)           
        
        
        if metadata['figure_tag']==True:
            Window_Number_i  #It will automaically gives you the results for the last window; however, you can change it here
            Figure_Table_Generator(Window_Number_i,learn_results,metadata)
        
    
        print("Done! \n \n")
 
    
        DF_results.to_pickle('output/DF_results'+metadata['model_type'][0]+'.pkl')  #This includes the same information as the html file but for all the test windows
        DF_Test_spacetime.to_pickle('output/DF_Test_spacetime'+metadata['model_type'][0]+'.pkl') #this includes the prediction probability for all methods and all test windows
        DF_Test_metric_time.to_pickle('output/DF_Test_metric_time'+metadata['model_type'][0]+'.pkl') #this includes the 4 metrics for each time window for all the test windows (the information used to draw figure 1)
        try:
            DF_results[DF_results['Window_Number']=='Mean'][['Window_Number','model','train_likelihood','test_likelihood','accuracy','precision','recall','f1','threshold','threshold_all_1','threshold_all_2','spearman_corr','pearson_corr']].to_excel('output/DF_results'+metadata['model_type'][0]+'_Mean-All-Window.xlsx',index=False)
        except:
            DF_results[DF_results['Window_Number']=='Mean'][['Window_Number','model','train_likelihood','test_likelihood','accuracy','precision','recall','f1','threshold','threshold_all_1','spearman_corr','pearson_corr']].to_excel('output/DF_results'+metadata['model_type'][0]+'_Mean-All-Window.xlsx',index=False)    
        
