"""
@Author- Ayan Mukhopadhyay
Compare count models and generate report.
Compares AIC and test data likelihood and summarizes as text file
"""

from jinja2 import Environment, FileSystemLoader
import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import configparser
import numpy as np

config = configparser.ConfigParser()
config.read("config/params.conf")














# Configure Jinja and ready the template
env = Environment(loader=FileSystemLoader(searchpath="reporting/templates"))
report_template = env.get_template("report_template.html")
regression_template = env.get_template("regression_template.html")
chart_template = env.get_template("chart_template.html")
table_template = env.get_template("table_template.html")


def ConfigSectionMap(config, section):
    """
    Reads the params.conf file into a dictionary
    @param config: parsed config
    @param section: section in the config file
    @return:
    """
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                print(("skip: %s" % option))
        except:
            print(("exception on %s!" % option))
            dict1[option] = None
    return dict1


def get_shorthand(model_name):
    """
    Simple dictionary lookup for shorthand names for regression models
    @param model_name: regression name
    @return: shorthand
    """
    try:
        return ConfigSectionMap(config, "shorthand")[model_name.lower()]
    except KeyError:
        return model_name


def build_regression_summary(df):
    """
    creates summary of regression results to be displayed in table
    @param results: dictionary of model names --> results
    @return: rendered table template
    """
    sections = list()
    for i,ROW in df.iterrows():
        #print(i)
        sections.append(table_template.render(
            model=ROW['Model'],
            cluster=ROW['Cluster'],
            train_l=ROW['Train Likelihood'],
            test_l=ROW['Test Likelihood'],
            predict_l=ROW['Prediction Likelihood'],
            test_mse=ROW['Test MSE'],
            predict_mse=ROW['Predict MSE'],    
            accuracy=ROW['Accuracy'],
            precision=ROW['Precision'],
            recall=ROW['Recall'], 
            f1=ROW['F1'],
            threshold=ROW['Threshold'],
            aic=ROW['AIC'],
            spearman_corr=ROW['Spearman'],
            pearson_corr=ROW['Pearson'],
            
        ))
    return sections




def build_regression_chart(data, param):
    """
    Parses the regression results and creates chart
    @param data: dataframe of results
    @param param: statistics name
    @return: rendered chart template
    """
    # df = pd.melt(df, id_vars="Model", var_name="Result", value_name="Likelihood")
    sns.set()
    sns_plot = sns.catplot(y='Model', x=param, data=data, hue="Cluster", kind='bar')
    sns_plot._legend.remove()
    sns_plot.set(ylabel=None);sns_plot.set(yticklabels=[])  # remove the tick labels  # remove the axis label to save space
    tmpfile = BytesIO()
    sns_plot.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    plt.close()
    return chart_template.render(encoded_img=encoded)


def generate_report(results,meta,Name):
    """
    Generate reports from regression results
    @param results: dictionary with model names and result statistics
    @return: _
    """
    for _,dict_value in results.items():
        #print(dict_value)
        for k, v in dict_value.items():
                try: 
                    dict_value[k] = round(v, 3)
                    #print(dict_value[k])
                except: 
                    try:
                        dict_value[k] = [round(i, 3) for i in v]
                        #print(dict_value[k])
                    except: #if (np.isnan(v)) | (v==None):
                        print(' ')
                        #print('No Data')

                
    '''
    v_1=np.nan
    np.isnan(v_1)
    
    v_2=None
    v_2==None
    
    if (np.isnan(v_1)) | (v_2==None):
        print('No Data')
    
    [round(i, 3) for i in v]  
        for _,dict_value in results.items():
            for k, v in dict_value.items():
                print(k,v)
    '''
    



    
    data = []
    metrics = ['Train Likelihood', 'Test Likelihood', 'Prediction Likelihood', 'Test MSE','Predict MSE' ,'Accuracy' , 'Precision', 'Recall', 'F1','Threshold' , 'AIC','Spearman','Pearson']
    for Counter, [k,v] in enumerate(results.items()):
        data.append([get_shorthand(k), 'All', v['train_likelihood'], v['test_likelihood'], v['predict_likelihood'], v['test_MSE'],
                                         v['predict_MSE'], v['accuracy'], v['precision'], v['recall'], v['f1'],v['threshold'], v['aic'],v['spearman_corr'],v['pearson_corr']])
        if meta[Counter]>1:
            for i in range(meta[Counter]):
                try:
                    data.append([get_shorthand(k), i, v['train_likelihood_all'][i], v['test_likelihood_all'][i], v['predict_likelihood_all'][i], v['test_MSE_all'][i],
                                             v['predict_MSE_all'][i], v['accuracy_all'][i], v['precision_all'][i], v['recall_all'][i], v['f1_all'][i], v['threshold_all'][i], v['aic_all'][i]])           
                except:
                    data.append([get_shorthand(k), i, np.nan, np.nan, np.nan, np.nan,
                                             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])           

                    
                    
        
    df = pd.DataFrame(data, columns=['Model']+['Cluster']+ metrics)

    imgs = []
    sections = build_regression_summary(df)
    for metric in metrics:
        imgs.append(build_regression_chart(data=df, param=metric))
    with open(Name, "w") as f:
        f.write(report_template.render(
                                        sections=sections,
                                        chart1=imgs[0],
                                        chart2=imgs[1],
                                        chart3=imgs[2],
                                        chart4=imgs[3],
                                        chart5=imgs[4],
                                        chart6=imgs[5],
                                        chart7=imgs[6],
                                        chart8=imgs[7],
                                        chart9=imgs[8],
                                        chart10=imgs[9],
                                        chart11=imgs[10],
                                        chart12=imgs[11],
                                        chart13=imgs[12]
        ))

