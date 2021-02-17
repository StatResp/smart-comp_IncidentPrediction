# KDD_IncidentPrediction
This repo is created for sharing the codes and data for the paper, Learning Incident Prediction Models Over Large GeographicalAreas for Emergency Response Systems, sumbitted to KDD



We have made the code used in this study publicly available here.   

## Instruction to run the file  
The following files are need to run the code. 
**config/params.conf**: This file contains metadata and parameters that must be customized to a user's specific deployment. This includes configuration information such as file paths and dataset information. It also configures which models to run, model hyperparameters, the features to use in regression, clustering parameters, and the synthetic sampling to apply. More detail on the specific metadata can be found in the readme file.
    
**run_sample.py**: This is the main script for fitting and evaluating the forecasting models. Based on the metadata in the **params.conf** file, this script loads and formats the data, calls sub-routines for clustering, synthetic sampling, tuning model specific hyper-parameters, and finally fit the desired models for each sliding test window. It then evaluates each model on a test set and outputs the following result files to the **output** directory: **DF_Results.pkl** - pandas dataframe which contains the overall evaluation metrics (accuracy, etc.) for each of the models averaged over space and time, **DF_Test_metric.pkl** - dataframe containing the evaluation metrics for each model on each 4 hour time window (aggregated over space), **DF_Test_spacetime.pkl** - dataframe containing the models' predictions for each 4-hour time window (used for resource allocation), **report.html** - html file which visualizes the evaluation results from the **DF_results** file. There is an example of the html results visualization provided in the repository. 
    
**run_allocation_multicore_v2.py**: This script evaluates the models’ impact when integrated with an allocation model. It uses the output from **run\_sample.py** as prediction inputs. Given a set of test incidents, the script performs allocation based on each model’s prediction output **DF_Test\_spacetime.pkl**, and then simulates dispatch to calculate the distance between incidents and their nearest responders. 


The data used in this study is proprietary, but we release a synthesized example dataset (in the **sample\data** folder) to demonstrate the expected data format. The data is formatted as a csv document; each row represents the features and incident counts for a 4 hour time window at a particular road segment. The specific feature names are detailed on the link provided. 
