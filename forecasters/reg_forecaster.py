"""
@Author - Sayyed Mohsen Vazirizade and Ayan Mukhopadhyay
Poisson Regression Forecaster -- Inherits from Forecaster class
"""

from forecasters.base import Forecaster
from forecasters.Resampling import Resampling_Func, Balance_Adjuster
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from copy import deepcopy
from scipy.special import factorial
from scipy import stats
from scipy.special import gamma
from sklearn.metrics import precision_recall_curve



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




class GLM_Model(Forecaster):
#This is the main GLM_Model calss, and all GLM fall into this category 
    def __init__(self,model_type):
        self.model_type = model_type
        self.model_params = {}
        self.model_stats = {}
        self.model_threshold= {}
        
        
    def __Threshold_Adjuster(self,model, x_verif, y_verif,temp_cluster):
            if self.model_type == 'LR':  
                #y_verif_hat=model.predict(df_cluster[~mask][metadata['features_ALL']])
                y_verif_hat= model.predict(x_verif)  
            elif self.model_type == 'ZIP'    :
                Model=GLM_Model_ZIP(model)
                x_verif['0_ZIP']=0 
                y_verif_hat = x_verif.apply(lambda x: 1-Model.get_probability_ZIP(x,x['0_ZIP']), axis=1)
                x_verif.drop('0_ZIP',axis=1)
             
            
            precision, recall, thresholds = precision_recall_curve(y_verif, y_verif_hat)
            fscore = (2 * precision * recall) / (precision + recall)
            ix = np.nanargmax(fscore)
            #ix = np.argmax(fscore)
            
            print('for Cluaster=%.0f, best Threshold=%f, F-Score=%.5f' % (temp_cluster, thresholds[ix], fscore[ix]))
            print('                        Threshold=%f, F-Score=%.5f' % (thresholds[ix-1], fscore[ix-1]))
            print('                        Threshold=%f, F-Score=%.5f' % (thresholds[ix+1], fscore[ix+1]))
            if np.isnan(fscore[ix])==True:
                self.model_threshold[temp_cluster] = 0.5
            else:
                self.model_threshold[temp_cluster] = thresholds[ix] 
                       

    def fit(self, df, metadata=None,resampling_type='No_Resample'):
        """
        Fits regression model to data
        @param df: dataframe of incidents in regression format
        @param metadata: dictionary with start and end dates, columns to regress on, cluster labels etc.
                        see github documentation for details
        @return: _
        """
        # if metadata is none, use standard parameters
        if metadata is None:
            metadata = create_default_meta(df)
        # get regression expression
        expr = self.get_regression_expr(metadata,metadata['features_ALL'])
        
        BalanceFactor=Balance_Adjuster(df,metadata)      #caclulate the Balance ratio between each cluster for resampling
        clusters = sorted(df.cluster_label.unique())
        for temp_cluster in clusters:
            print('temp_cluster',temp_cluster)
            df_cluster = df.loc[df.cluster_label == temp_cluster]
            Y, X = dmatrices(expr, df_cluster, return_type='dataframe')
            #Unbalanced_Correction=0
            #Unbalanced_Correction_Type='SMOTE'
            np.random.seed(seed=0)
            split_point = metadata['train_verification_split']
            mask=df_cluster.index.isin(df_cluster.sample(int(split_point*len(df_cluster)),replace=False).index)   
            x_train=X[mask]
            y_train=Y[mask]
            x_verif=X[~mask]
            y_verif=Y[~mask]        
            
            if resampling_type in ['RUS','ROS','SMOTE']:
                sampling= Resampling_Func(resampling_type,BalanceFactor[temp_cluster] )
                if metadata['current_model']['model_type'] in metadata['TF_Models']:
                    x_train,y_train = sampling.fit_resample(x_train,y_train)
                elif metadata['current_model']['model_type'] in metadata['Count_Models']:                
                        print('Warning: prediction column is not binary for resampling')
                        x_train,_ = sampling.fit_resample(x_train,y_train>0)
                        y_train,_ = sampling.fit_resample(y_train,y_train>0)
        
            # fit model
            if (self.model_type == 'PR'): #Poisson_Regression
                model = sm.GLM(y_train, x_train, family=sm.families.Poisson(sm.families.links.log())).fit()
            elif self.model_type == 'LR':  #Logistic_Regression
                                
                model = sm.GLM(y_train, x_train, family=sm.families.Binomial(sm.families.links.logit())).fit(method="lbfgs")  #lbfgs #bfgs irls
                self.__Threshold_Adjuster(model, x_verif, y_verif,temp_cluster ) 
                
            elif self.model_type == 'SR':     #Simple_Regression
                try:
                    model = sm.GLM(y_train, x_train, family=sm.families.Gaussian(sm.families.links.identity())).fit()  #the the dependent variable is exactly the same as one of the features, it will run into an error
                except:
                     model = sm.OLS(y_train, x_train).fit()
            elif (self.model_type == 'NBR'):   #Negative_Binomial_Regression
                model = sm.GLM(y_train, x_train, family=sm.families.Poisson(sm.families.links.log())).fit()    
                y_train_predicted=deepcopy(y_train)
                y_train_predicted.columns=['y']
                y_train_predicted['p']=model.predict(x_train)         
                y_train_predicted['error']=(y_train_predicted['y']-y_train_predicted['p'])
                ancillary = sm.OLS(y_train_predicted['error']**2-y_train_predicted['p'], y_train_predicted['p']**2).fit().params[0]  
                try:
                    model=sm.GLM(y_train, x_train, family=sm.families.NegativeBinomial(alpha=ancillary)).fit()
                except:
                    print('Warning! ancilalary is considered 1')
                    model=sm.GLM(y_train, x_train, family=sm.families.NegativeBinomial(alpha=1)).fit()  
            elif (self.model_type == 'ZIP'):  #Zero_Inflated_Poisson_Regression
                print('===================')
                model = sm.ZeroInflatedPoisson(endog=y_train, exog=x_train, exog_infl=x_train, inflation='logit').fit(method="lbfgs",maxiter=500,full_output=True,disp=True)  #method="nm" method='bfgs' lbfgs    #https://www.statsmodels.org/stable/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.fit.html
                #model = sm.ZeroInflatedPoisson(endog=y_train, exog=x_train, inflation='logit').fit(method="bfgs",maxiter=200) #method="nm"
                print('But did it converge? checking the mle_retvals: ', model.mle_retvals["converged"])
                print('===================')
                print('\n \n') 
                
                self.__Threshold_Adjuster(model, x_verif, df_cluster[~mask][metadata['pred_name_TF']],temp_cluster ) 
                #self.model_threshold[temp_cluster] = 0.5
                print(model.summary())
            self.model_params[temp_cluster] = model     
        self.update_model_stats()
        print('Finished Learning {} model'.format(self.model_type))

    def prediction(self, df, metadata):
        """
        Predicts E(y|x) for a set of x, where y is the concerned dependent variable.
        @param df: dataframe of incidents in regression format
        @param metadata: dictionary with start and end dates, columns to regress on, cluster labels etc.
                        see github documentation for details
        @return: updated dataframe with predicted values, Sigma2, ancillary and information regarding llf and MSE
        """
        df_complete_predicted=df.copy()
        df_complete_predicted['predicted']=0
        df_complete_predicted['Sigma2']=0
        df_complete_predicted['ancillary']=0
        features = metadata['features_ALL']
        
        clusters = sorted(df.cluster_label.unique())
        for temp_cluster in clusters:
            print('temp_cluster',temp_cluster)
            #df_cluster = df.loc[df.cluster_label == temp_cluster]
            if (self.model_type == 'ZIP'):
                Model=GLM_Model_ZIP(self.model_params[temp_cluster])
                df_complete_predicted.loc[df.cluster_label == temp_cluster,'predicted_Count'] = Model.model.predict( df[df.cluster_label == temp_cluster][features],exog_infl= df[df.cluster_label == temp_cluster][features])
                df['0_ZIP']=0 
                df_complete_predicted.loc[df.cluster_label == temp_cluster,'predicted'] = df[df.cluster_label == temp_cluster].apply(lambda x: 1-Model.get_probability_ZIP(x,x['0_ZIP']), axis=1)
                df.drop('0_ZIP',axis=1)
                ###df_complete_predicted.loc[df.cluster_label == temp_cluster,'predicted'] = Model.model_params[temp_cluster].predict( df[df.cluster_label == temp_cluster][features],exog_infl= np.ones((len(df[df.cluster_label == temp_cluster][features]),1)))
            
            else:
                #df_cluster['predicted'] = self.model_params[temp_cluster].predict(df_cluster[features])
                df_complete_predicted.loc[df.cluster_label == temp_cluster,'predicted'] =  self.model_params[temp_cluster].predict(df[df.cluster_label == temp_cluster][features])
            #df_cluster['predicted'].replace(to_replace=0, value=1/1e8)
            #df_complete_predicted['predicted'].replace(to_replace=0, value=1/1e8)
            #df_complete_predicted.loc[df.cluster_label==temp_cluster,'predicted']=deepcopy(df_cluster['predicted'])
            df_complete_predicted.loc[df.cluster_label==temp_cluster,'Sigma2']=self.model_params[temp_cluster].scale/self.model_params[temp_cluster].nobs*self.model_params[temp_cluster].df_resid
            if (self.model_type == 'LR') | (self.model_type == 'ZIP'):  
                df_complete_predicted.loc[df.cluster_label==temp_cluster,'threshold']=self.model_threshold[temp_cluster]
            else :  
                df_complete_predicted.loc[df.cluster_label==temp_cluster,'threshold']=0.5
                
        #if self.model_type == 'Logistic_Regression':  
                ##df_complete_predicted['predicted_TF']=df_complete_predicted['predicted']>0.5
                #df_complete_predicted['predicted_TF']=df_complete_predicted['predicted']>df_complete_predicted['threshold']
        df_complete_predicted['predicted_TF']=df_complete_predicted['predicted']>df_complete_predicted['threshold']
            
        if (metadata['pred_name_TF'] in  df.columns): #| (metadata['pred_name_Count'] in  df.columns):
            df_complete_predicted['error']=df_complete_predicted[metadata['pred_name_TF']]-df_complete_predicted['predicted']
            for temp_cluster in clusters:
                df_complete_predicted.loc[df.cluster_label==temp_cluster,'ancillary']=   sm.OLS(df_complete_predicted.loc[df.cluster_label==temp_cluster,'error']**2-
                                                                                                df_complete_predicted.loc[df.cluster_label==temp_cluster,'predicted'],
                                                                                                df_complete_predicted.loc[df.cluster_label==temp_cluster,'predicted']**2).fit().params[0]    
            test_likelihood_all,  test_likelihood, df = self.Likelihood(df_complete_predicted, metadata)
            df_complete_predicted['llf']=df['llf']
            MSE_all,  MSE = self.MSE(df_complete_predicted, metadata)     
            return [df_complete_predicted,test_likelihood_all,test_likelihood,MSE_all, MSE]
        else:
            return [df_complete_predicted,None,None,None]
        
    def get_regression_expr(self, metadata,features):
        """
        Creates regression expression in the form of a patsy expression
        @param features: x features to regress on
        @return: string expression
        """
        if metadata['current_model']['model_type'] in metadata['TF_Models']:
            expr = metadata['pred_name_TF']+'~'
        elif metadata['current_model']['model_type'] in metadata['Count_Models']:
            expr = metadata['pred_name_Count']+'~'
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

    def Likelihood(self, df, metadata):
        #smv:checked
        """
        Return the likelihood of model for the incidents in df for prediction
        @df: dataframe of incidents to calculate likelihood on
        @param metadata: dictionary with feature names, cluster labels.
        @return: likelihood value for each sample and the total summation as well the updated df which includes llf
        """ 
        
        # fit model
        if self.model_type == 'PR':
            df['llf']=np.log(stats.poisson.pmf(k=df[metadata['pred_name_Count']], mu=df['predicted']))
        if self.model_type == 'LR':
            df['llf']=np.log(stats.bernoulli.pmf(k=df[metadata['pred_name_TF']], p=df['predicted']))
        elif self.model_type == 'SR':
            #df['llf']=np.log(stats.norm.pdf(df[metadata['pred_name_Count']], df['predicted'],self.model_params[df['cluster_label']].scale)    )
            df['llf']=np.log(stats.norm.pdf(df[metadata['pred_name_Count']], df['predicted'],df['Sigma2']**0.5)    )
        elif self.model_type == 'NBR':
            df['llf']=np.log(self.get_likelihood_NB(df,metadata))
        elif self.model_type == 'ZIP':
            df['llf']=np.log(self.get_likelihood_ZIP(df,metadata,metadata['pred_name_Count']))
            
        test_likelihood_all=df[['llf','cluster_label']].groupby(['cluster_label'], as_index=False).sum().sort_values(by='cluster_label', ascending=True)
        test_likelihood=sum(test_likelihood_all['llf'])
        return [(test_likelihood_all['llf'].values).tolist(),test_likelihood,df]
    
    
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
    

    def update_model_stats(self):
        #smv:checked
        """
        Store the the summation of log likelihood of the training set, AIC value.
        @return: _
        """
        train_likelihood = []
        aic = []
        for temp_cluster in self.model_params.keys():
            train_likelihood.append(self.model_params[temp_cluster].llf )  #llf: Value of the loglikelihood function evalued at params.
            aic.append( self.model_params[temp_cluster].aic)

        self.model_stats['train_likelihood'] = sum(train_likelihood)
        self.model_stats['aic'] = sum(aic)
        self.model_stats['train_likelihood_all']=train_likelihood
        self.model_stats['aic_all']=aic



    def get_likelihood_NB(self, df,metadata):
        """
        Return the likelihood of NB model for the incidents in df for prediction
        @df: dataframe of incidents to calculate likelihood on
        @return: list of likelihood values for each sample
        """ 
    
        method=2
        if method==1:
        ##method1:   
            l_temp=(stats.nbinom.pmf(df[metadata['pred_name_TF']],
                                     1/df['ancillary'],
                                 1   /(1+df['predicted']*df['ancillary'])))
        if method==2:
        ##method2:            
            l_temp = gamma(df[metadata['pred_name_TF']] + 1/df['ancillary']) / (gamma(1/df['ancillary']) * gamma(df[metadata['pred_name_TF']]+1))
            l_temp *= (1 / (1 + df['ancillary'] * df['predicted'])) ** (1/df['ancillary'])
            l_temp *= ((df['ancillary'] * df['predicted']) / (1 + df['ancillary'] * df['predicted'])) ** df[metadata['pred_name_TF']]

        
        return l_temp
        
        
    def get_likelihood_ZIP(self, df,metadata, pred_name_col):
        clusters = df.cluster_label.unique()
        for temp_cluster in clusters:
            Model=GLM_Model_ZIP(self.model_params[temp_cluster])
            df.loc[df.cluster_label==temp_cluster,'llf']=df[df.cluster_label == temp_cluster].apply(lambda x: Model.get_probability_ZIP(x,x[pred_name_col]), axis=1)
        return df['llf']    
    





class GLM_Model_ZIP(GLM_Model):
#This is the main GLM_Model calss, and all GLM fall into this category 
    def __init__(self,model):
        self.model=model
        
    def get_probability_ZIP(self, x,y): 

        """
        Return the likelihood of ZIP model for the incidents in df for prediction
        @df: dataframe of incidents to calculate likelihood on
        @return: list of likelihood values for each sample
        """ 
    
        """
        Calculate the likelihood of a zero-inflated poisson model. The likelihood of the inflated model is calculated
        through coefficients marked with 'inflate_'
        @param df: dataframe whose likelihood needs to be calculated
        @param param: set of regression coefficients
        @return: likelihood value
        """
        
        
        sigmoid = lambda x: 1 / (1 + np.exp(-1 * x))
        
        def parse_inflate_model(coef):
            """
            Seperate the coefficients for inflated part and the Poission regression part through coefficients marked with 'inflate_'
            @param coef: all the coefficients
            @return: coef_inflate and coef_reg which represent the coefficients for inflated part and the Poission regression part, respectively
            """            
            coef_reg = {}
            coef_inflate = {}
            for k,v in coef.items():
                if 'inflate_' in k:
                    coef_inflate[k] = v
                else:
                    coef_reg[k] = v
            return coef_reg, coef_inflate        
        
        
        
        def lik_calc(x,y,coef_reg, coef_inflate ):
            """
            calculate the likelihood for each sample point
            @param x: a row of the incdient df which includes the features and predicted values
            @return: likelihood
            """              
            
            
            temp_inflate = 0
            temp_poisson = 0
    
            for key, val in coef_inflate.items():
                feature = key.split('inflate_')[1]  # retrieve the part after "inflate_"
                temp_inflate += coef_inflate[key] * x[feature]
    
            for key, val in coef_reg.items():
                try:
                    temp_poisson += coef_reg[key] * x[key]
                except KeyError:
                    # if intercept is not in the features
                    temp_poisson += coef_reg[key] * 1
    
            p_lambda = np.exp(temp_poisson)
            p_logistic = sigmoid(temp_inflate)
    
            if y== 0:  # use embedded inflated model (logistic regression usually)
                L = p_logistic + (1-p_logistic) * np.exp(-1 * p_lambda)
    
            else:  # use embedded inflated model for P(y|x) != 0 * P(count=y|x)
                L = (1 - p_logistic) *  (p_lambda ** y) * np.exp(-1 * p_lambda) / np.math.factorial(y)
            return L
        
        coef_reg, coef_inflate = parse_inflate_model(self.model.params)
        L=lik_calc(x,y,coef_reg, coef_inflate )
        return L
    