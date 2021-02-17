# KDD_IncidentPrediction
This repo is created for sharing the codes and data for the paper, Learning Incident Prediction Models Over Large GeographicalAreas for Emergency Response Systems, sumbitted to KDD. Figure below shows the general pipeline designed by authors. 

![pipeline_updated](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_in_Paper/pipeline_updated.png)

We have made the code used in this study publicly available here.   

## Instruction to run the file  
The following files are need to run the code. 
**config/params.conf**: This file contains metadata and parameters that must be customized to a user's specific deployment. This includes configuration information such as file paths and dataset information. It also configures which models to run, model hyperparameters, the features to use in regression, clustering parameters, and the synthetic sampling to apply. More detail on the specific metadata can be found in the readme file.
    
**run_sample.py**: This is the main script for fitting and evaluating the forecasting models. Based on the metadata in the **params.conf** file, this script loads and formats the data, calls sub-routines for clustering, synthetic sampling, tuning model specific hyper-parameters, and finally fit the desired models for each sliding test window. It then evaluates each model on a test set and outputs the following result files to the **output** directory: **DF_Results.pkl** - pandas dataframe which contains the overall evaluation metrics (accuracy, etc.) for each of the models averaged over space and time, **DF_Test_metric.pkl** - dataframe containing the evaluation metrics for each model on each 4 hour time window (aggregated over space), **DF_Test_spacetime.pkl** - dataframe containing the models' predictions for each 4-hour time window (used for resource allocation), **report.html** - html file which visualizes the evaluation results from the **DF_results** file. There is an example of the html results visualization provided in the repository. Based on the settings, the code can generate heatmap data as well, which are saved in html format. 
    
**run_allocation_multicore_v2.py**: This script evaluates the models’ impact when integrated with an allocation model. It uses the output from **run\_sample.py** as prediction inputs. Given a set of test incidents, the script performs allocation based on each model’s prediction output **DF_Test\_spacetime.pkl**, and then simulates dispatch to calculate the distance between incidents and their nearest responders. 


**Reading_All_DF.py**: If the analyses is done in seperate environment, this script brings all of them together and save them in one file. For example, the analyses for RF, NN, LR, and ZIP can be done in seperate environment. This script combines all **DF_Results.pkl**, **DF_Test_spacetime.pkl**, and **DF_Test_metric.pkl** from various analyses abd builds **DF_Results_LR+NN+RF+ZIP.pkl**, **DF_Test_spacetime_LR+NN+RF+ZIP.pkl**, and **DF_Test_metric_LR+NN+RF+ZIP.pkl**.  It also creates **Perfromance.xlsx**, based of which table 2 in the paper is generated. 


**Reading_All_Distances.py**: This script collects the output of **run_allocation_multicore_v2.py** and saves them in **Distance.pkl**. This script also creates **AllcationResults.xlsx**, based of which table 3 and figure 7 in the paper are generated. 



**anonymous_df.pkl**: The data used in this study is proprietary, but we release a synthesized example dataset (in the **sample/data** folder) to demonstrate the expected data format. The data is formatted as a csv document; each row represents the features and incident counts for a 4 hour time window at a particular road segment. The specific feature names are detailed on the link provided. 



## Results of the Paper
**results_KDD** includes all the results, tables, and high resolution figures based on which the paper is written. 

For the sake of illustration, besideds the actual data, different spatio termporal predictions for Jan/2020 are shown in the following section based on Logistic regression, different resampling techniques, and with and without resampling. The figure below demonstrates the general lay out of all the figures. 
![lay_out](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/lay_out.png)  

\*blue pixels show the missing values.  


0)  
![Actual Data](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_testwindow(0)_Actual Data.png)
1)  
![LR+NoR+KM2](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+NoR+KM2_testwindow(0)_Prediction.png)
2)  
![LR+NoR+NoC1](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+NoR+NoC1_testwindow(0)_Prediction.png)
3)  
![LR+ROS+KM2](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+ROS+KM2_testwindow(0)_Prediction.png)
4)  
![LR+ROS+NoC1](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+ROS+NoC1_testwindow(0)_Prediction.png)
5)  
![LR+RUS+KM2](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+RUS+KM2_testwindow(0)_Prediction.png)
6)  
![LR+RUS+NoC1](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+RUS+NoC1_testwindow(0)_Prediction.png)





