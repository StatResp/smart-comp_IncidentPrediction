"""
@Author - Sayyed Mohsen Vazirizade and Ayan Mukhopadhyay
Parent File for all regression models for incident prediction.
Currently supports -
1. Poisson Regression (Count Based)
2. Negative Binomial Regression (Count Based)
3. Parametric Survival Regression (Time Based)
"""
from forecasters.reg_forecaster import GLM_Model
from forecasters.non_linear_forecaster import SVM
from forecasters.utils import  update_meta_features
import numpy as np
from pprint import pprint
from clustering.simple_clustering import Simple_Cluster
import pandas as pd



def learn(df, metadata, current_model):
    """
    Wrapper before data is passed to specific regression models. Converts raw incident data to a format that regression
    models can use. For example, creates counts per time period for Poisson model. Splits the data into train and test
    sets for cross-validation.
    @param df: raw dataframe of incidents
    @param metadata: metadata with start and end dates, spatial unit etc. See github documentation for details
    @param model_type: the name of the regression model
    @return: trained model and regression df
                               regression df: 
                                             df_train and df_test
                                             df_predict                         
    """
    #count_models = ['Simple_Regression','Logistic_Regression','Poisson_Regression', 'Negative_Binomial_Regression','Zero_Inflated_Poisson_Regression']

    #Random number seed
    np.random.seed(seed=0)
    
    # create dataframe for regression
    
    
    
    def create_train_test_predict(df, metadata):
        """
        Splits the dataframe in to train, test, and predict.
          
        @param df: raw dataframe
        @param train: This the dataframe we train the model based on
        @param test: The model will never see this data in the prediction phase
        @param predict: This is just for the sake of illustartion mainly. This dataframe is used to for prediction when the model is trained. 
        """
        #if df_regression has already been built, we dont have to spend time and make it again 
        
        #df_regression = pickle.load(open(metadata['regressiondf_pickle_address'], 'rb'))
        # split into train and predict data sets (df.time >= window_start) & (df.time < window_end)
        #df_train   = df_regression.loc[(df_regression['time'] >= metadata['start_time_train'])   &   (df_regression['time'] < metadata['end_time_train']) ]
        #df_predict = df_regression.loc[(df_regression['time'] >= metadata['start_time_predict'])    &   (df_regression['time'] < metadata['end_time_predict']) ]
        print('Attn! The file for regressiondf exists so it will not be created again!')
        
        if (metadata['train_test_type']=='simple') | (metadata['train_test_type']=='moving_window'):
            print('The train, test, and predict are defined just based on the input dates')
            df_train   = df.loc[(df[metadata['time_column']] >= metadata['start_time_train'])   &   (df[metadata['time_column']] < metadata['end_time_train']) ]
            df_test   = df.loc[(df[metadata['time_column']] >= metadata['start_time_test'])   &   (df[metadata['time_column']] < metadata['end_time_test']) ]
            df_predict = df.loc[(df[metadata['time_column']] >= metadata['start_time_predict'])    &   (df[metadata['time_column']] < metadata['end_time_predict']) ]
            
        elif metadata['train_test_type']=='random_speudo':   
            
            df_learn   = df.loc[(df[metadata['time_column']] >= metadata['start_time_train'])   &   (df[metadata['time_column']] < metadata['end_time_train']) ]
            df_predict = df.loc[(df[metadata['time_column']] >= metadata['start_time_predict'])    &   (df[metadata['time_column']] < metadata['end_time_predict']) ]
            
            Week_2018=(((df_learn['time_local'].dt.week==19) | (df_learn['time_local'].dt.week==30)) & (df_learn['time_local'].dt.year==2018))
            Week_2019=(((df_learn['time_local'].dt.week==19) | (df_learn['time_local'].dt.week==31)) & (df_learn['time_local'].dt.year==2019))
            mask= (Week_2018 | Week_2019)==False
            
            df_train = df_learn[mask]
            df_test = df_learn[~mask]



        elif metadata['train_test_type']=='ratio_random':   
            
            df_learn   = df.loc[(df[metadata['time_column']] >= metadata['start_time_train'])   &   (df[metadata['time_column']] < metadata['end_time_train']) ]
            df_predict = df.loc[(df[metadata['time_column']] >= metadata['start_time_predict'])    &   (df[metadata['time_column']] < metadata['end_time_predict']) ]
            

            split_point = metadata['train_test_split']
            mask=df_learn.index.isin(df_learn.sample(int(split_point*len(df_learn)),replace=False).index)
            #mask = np.random.rand(len(df_learn)) < split_point    #since by using this method, the length of the df_learn may change, the above method is prefered 
            
            df_train = df_learn[mask]
            df_test = df_learn[~mask]            
            
            
            
            
            
        #return {'train': df_train, 'test': df_test, 'predict':df_predict} 
        df_train=df_train.reset_index()
        df_test=df_test.reset_index()
        df_predict=df_predict.reset_index()
        return  df_train, df_test, df_predict



    def Train_Verification_Split(df_learn,metadata):
        #For now I just include a simple random method, but we can use other methods like cross validation later if it is needed
        
            split_point = metadata['train_verification_split']
            mask=df_learn.index.isin(df_learn.sample(int(split_point*len(df_learn)),replace=False).index)        
        
            df_train = df_learn[mask]
            df_verif = df_learn[~mask]   
            return df_train,df_verif    
    
    #regression_data = create_regression_df(df, metadata, model_name)
    #df_learn = regression_data['train']
    #df_predict = regression_data['predict']
    ##updating the features based on categorical ones and adding the intercept
    ##metadata['features_ALL'] = update_meta_features(metadata['features_ALL'], df_features=list(df_learn.columns), cat_col=metadata['cat_features'])
    #metadata['features_ALL'] = update_meta_features(metadata['features_ALL'], df_features=df_learn.columns.tolist(), cat_col=metadata['cat_features'])
    
    df_train, df_test, df_predict= create_train_test_predict(df, metadata)
    metadata['features_ALL'] = update_meta_features(metadata['features_ALL'], df_features=df_train.columns.tolist(), cat_col=metadata['cat_features'])
        
   
    if metadata['current_model']['cluster_type']=="KM":
        Cluster=Simple_Cluster('kmeans',metadata['current_model']['cluster_number'])
        df_train, df_test, df_predict=      Cluster.fit_pred(df_train,df_test,df_predict, metadata)
    else:
        print('No Clustering')
        #df_learn['cluster_label']=0
        df_train.loc[:,'cluster_label'] = 0  
        #df_verif.loc[:,'cluster_label'] = 0
        df_test.loc[:,'cluster_label'] = 0  
        df_predict.loc[:,'cluster_label'] = 0  

    #df_train, df_verif=Train_Verification_Split(df_learn,metadata) 
    
    
    '''
    Unbalanced_Correction=0
    Unbalanced_Correction_Type='RUS'
    if Unbalanced_Correction==1:
        if Unbalanced_Correction_Type=='RUS':        
            from imblearn.under_sampling import RandomUnderSampler
            sampling = RandomUnderSampler(sampling_strategy='majority',random_state=1)
        elif Unbalanced_Correction_Type=='ROS':        
            from imblearn.over_sampling import RandomOverSampler
            sampling = RandomOverSampler(sampling_strategy='minority',random_state=1)
        elif Unbalanced_Correction_Type=='SMOTE':
            from imblearn.over_sampling import SMOTE
            #from imblearn.over_sampling import BorderlineSMOTE
            sampling = SMOTE(sampling_strategy='not majority',random_state=1)
            #oversample = BorderlineSMOTE()
        df_train,_ = sampling.fit_resample(df_train,df_train[metadata['pred_name']])
        #df_train_sample,_ = sampling.fit_resample(df_train[metadata['features_ALL']],df_train[metadata['pred_name']])
    '''
    
    
    
    #print('The updated features are:', metadata['features_ALL'])
    #if (model_name == 'Poisson_Regression') | (model_name == 'Logistic_Regression') | (model_name == 'Simple_Regression') | (model_name == 'Negative_Binomial_Regression') | (model_name == 'Zero_Inflated_Poisson_Regression')  :
    if metadata['current_model']['model_type'] in metadata['GLM_Models']:
        model = GLM_Model(metadata['current_model']['model_type'])
        model.fit(df_train, metadata,metadata['current_model']['resampling_type'])
    elif metadata['current_model']['model_type']== 'Survival_Regression':
        model = Survival_Model()
        model.fit(df_train, metadata)
    elif metadata['current_model']['model_type']== 'SVM':
        model = SVM(metadata['current_model']['model_type'])
        model.fit(df_train, metadata)
        
    elif metadata['current_model']['model_type'] == 'NN':
        model = NN(metadata['current_model']['model_type'])
        model.fit(df_train, metadata)

    elif metadata['current_model']['model_type']== 'RF':
        model = RF(metadata['current_model']['model_type'])
        model.fit(df_train, metadata)
                       

    return {'model':model, 'df_train':df_train, 'df_test':df_test, 'df_predict':df_predict}


