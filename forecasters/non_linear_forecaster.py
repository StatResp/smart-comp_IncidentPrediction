"""
@Author - Ayan Mukhopadhyay
Classification Forecaster -- Inherits from Forecaster class
"""

from forecasters.base import Forecaster
import numpy as np
import pandas as pd
from copy import deepcopy
from patsy import dmatrices
from sklearn import svm
from sklearn import metrics
from forecasters.Resampling import Resampling_Func, Balance_Adjuster
import time

def create_default_meta(df, static_features=None):
    """
    Creates default set of metadata if user supplied data is missing
    @param df: dataframe of incidents
    @param static_features: set of static features used in clustering
    @return: metadata dictionary
    """
    metadata = {'start_time_train': df['time'].min(), 'end_time_train': df['time'].max()}
    if static_features is None:
        static_features = list(df.columns)
        if 'cluster_label' in static_features:
            static_features.remove('cluster_label')
    metadata['features_ALL'] = static_features
    return metadata


class SVM(Forecaster):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model_params = {}
        self.model_stats = {}

    def fit(self, df, metadata=None, sampling_name='No_Resample', kernel='rbf'):            #kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
        # if metadata is none, use standard parameters
        if metadata is None:
            metadata = create_default_meta(df)

        # get regression expression
        expr = self.get_regression_expr(metadata['features_ALL'])
        clusters = df.cluster_label.unique()
        
        
        BalanceFactor=Balance_Adjuster(df,metadata)      #caclulate the Balance ratio between each cluster for resampling
        clusters = sorted(df.cluster_label.unique())
        for temp_cluster in clusters:
            print('temp_cluster',temp_cluster)
            df_cluster = df.loc[df.cluster_label == temp_cluster]
            #y_train, x_train = dmatrices(expr, df_cluster.iloc[0:100000], return_type='dataframe')
            y_train, x_train = dmatrices(expr, df_cluster, return_type='dataframe')
            if kernel is None:
                model = svm.SVC()  # default kernel is rbf
            else:
                model = svm.SVC(kernel=kernel)            
            #Unbalanced_Correction=0
            #Unbalanced_Correction_Type='SMOTE'
            if sampling_name in ['RUS','ROS','SMOTE']:
                sampling= Resampling_Func(sampling_name,BalanceFactor[temp_cluster] )
                x_train,y_train = sampling.fit_resample(x_train,y_train)        
        
            #model.fit(x_train.iloc[0:100], y_train.iloc[0:100])
            #model.fit(np.asarray(x_train), np.asarray(y_train).values.ravel())
            #model.fit(x_train, y_train.values.ravel())
            start_time = time.time()
            model.fit(x_train.values, y_train.values.ravel())
            print("--- %s seconds for running SVM---" % (time.time() - start_time))

            #model.fit(x_train, y_train)
            pred = model.predict(x_train)
            self.model_params[temp_cluster] = model
            self.update_model_stats(y_train, pred)

    def update_model_stats(self, y_train, pred):
        # smv:checked
        """
        Store the the summation of log likelihood of the training set, AIC value.
        @return: _
        """
        train_likelihood = []
        aic = []
        for temp_cluster in self.model_params.keys():
            train_likelihood.append(np.nan )  #llf: Value of the loglikelihood function evalued at params.
            aic.append(np.nan)

        self.model_stats['train_likelihood'] = np.nan
        self.model_stats['aic'] = np.nan
        self.model_stats['train_likelihood_all']=np.nan
        self.model_stats['aic_all']=np.nan

        
        


    def prediction(self, df, metadata):
        """
        Predicts E(y|x) for a set of x, where y is the concerned dependent variable.
        @param df: dataframe of incidents in regression format
        @param metadata: dictionary with start and end dates, columns to regress on, cluster labels etc.
                        see github documentation for details
        @return: updated dataframe with predicted values and information about the MSE
        """
        df_complete_predicted = df
        df_complete_predicted['predicted'] = 0
        df_complete_predicted['Sigma2'] = 0
        df_complete_predicted['ancillary'] = 0
        features = metadata['features_ALL']

        clusters = df.cluster_label.unique()
        for temp_cluster in clusters:
            df_cluster = df.loc[df.cluster_label == temp_cluster]
            df_cluster['predicted'] = self.model_params[temp_cluster].predict(df_cluster[features])
            print('temp_cluster ', temp_cluster)

            df_complete_predicted.loc[df.cluster_label == temp_cluster, 'predicted'] = deepcopy(df_cluster['predicted'])

        if metadata['pred_name'] in df_cluster.columns:
            df_complete_predicted['error'] = df_complete_predicted[metadata['pred_name']] - df_complete_predicted['predicted']
            MSE_all,  MSE = self.MSE(df_complete_predicted, metadata)     
            return [df_complete_predicted, None, None,MSE_all, MSE]
        else:
            return [df_complete_predicted, None, None, None]
        
        
        
        
    def get_regression_expr(self, features):
        """
        Creates regression expression in the form of a patsy expression
        @param features: x features to regress on
        @return: string expression
        """
        expr = "count~"
        for i in range(len(features)):
            # patsy expects 'weird columns' to be inside Q
            if ' ' in features[i]:
                expr += "Q('" + features[i] + "')"
            else:
                expr += features[i]
            if i != len(features) - 1:
                expr += '+'
        expr  += '-1'
        return expr



    def MSE(self, df, metadata):
        #smv:checked
        """
        Return the Mean Square Error (MSE) of model for the incidents in df for prediction
        @df: dataframe of incidents to calculate likelihood on
        @param metadata: dictionary with feature names, cluster labels.
        @return: likelihood value for each sample and the total summation as well the updated df which includes llf
        """ 
        df['error2']=df['error']**2
        MSE=np.mean(df['error2'])                 
        MSE_all=df[['error2','cluster_label']].groupby(['cluster_label'], as_index=False).mean().sort_values(by='cluster_label', ascending=True)
        return [(MSE_all['error2'].values).tolist(),MSE]
