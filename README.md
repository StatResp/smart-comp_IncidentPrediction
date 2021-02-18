# KDD_IncidentPrediction
This repo is created for reproducibility and sharing the codes used for the paper, *Learning Incident Prediction Models Over Large Geographical Areas for Emergency Response Systems*, submitted to KDD 2021. Figure below shows the general pipeline designed by authors.
<img src="https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_in_Paper/pipeline_updated.png" width="800">  

Firstly, we explain the code and its required input and generated output. Since the authors did not have the permission to share the original data, a synthetic dataset based on a sector of the original data is created, which allow the users to run the code and see the output. Then, we show some abilities of the pipeline, which are not mentioned in the paper due to space limitation.


## Instruction to run the file  

### Prediction
**config/params.conf**: This file contains metadata and parameters that must be customized to a user's specific deployment. This includes configuration information such as file paths and dataset information. It also configures which models to run, model hyperparameters, the features to use in regression, clustering parameters, and the synthetic sampling to apply. In summary, anything that is not data but the pipeline needs, should be incorporated here. **params.conf**** can be used for synthetic data. **params_LR.conf** also included in **config** folder. This is the file we used to generate the results for various logistic regression models. 
    
**run_sample.py**: This is the main script for fitting and evaluating the forecasting models. Based on the metadata in the **params.conf** file, this script loads and formats the data, calls sub-routines for clustering, synthetic sampling, tuning model specific hyper-parameters, and finally fit the desired models for each sliding test window. It then evaluates each model on a test set and outputs the following result files to the **output** directory: **DF_Results.pkl** - pandas dataframe which contains the overall evaluation metrics (accuracy, etc.) for each of the models averaged over space and time, **DF_Test_metric.pkl** - dataframe containing the evaluation metrics for each model on each 4 hour time window (aggregated over space), **DF_Test_spacetime.pkl** - dataframe containing the models' predictions for each 4-hour time window (used for resource allocation), **report.html** - html file which visualizes the evaluation results from the **DF_results** file. Based on the settings, the code can generate spatio temporal prediction figures and heatmap figures. 
    

### Allocation
**run_allocation_multicore_v2.py**: This script evaluates the models’ impact when integrated with an allocation model. It uses the output from **run\_sample.py** as prediction inputs. Given a set of test incidents, the script performs allocation based on each model’s prediction output **DF_Test\_spacetime.pkl**, and then simulates dispatch to calculate the distance between incidents and their nearest responders. 

**Reading_All_Distances.py**: This script collects the output of **run_allocation_multicore_v2.py** and saves them in **Distance.pkl**. This script also creates **AllcationResults.xlsx**, based of which table 3 and figure 7 in the paper are generated. 


### Sample Analysis

**anonymous_df.pkl**: The data used in this study is proprietary, but we release a synthesized example dataset (in the **sample/data** folder) to demonstrate the expected data format. The data is formatted as a csv document; each row represents the features and incident counts for a 4-hour time window at a particular road segment. The specific feature names are detailed on the link provided. By Using **anonymous_df.pkl** and **config/params.conf** and running **run_sample.py** the results will be saved in **output** folder. For the sake of reproducibility, the authors ran the model and saved the results in the same folder. The user should be able to reproduce the same results. The original data used for the paper can be shared by format request and under specific conditions. 



## Results of the Paper
**results_KDD** includes all the results, tables, and high resolution figures based on which the paper is written. Below, also we show some abilities of the pipeline, which are not mentioned in the paper due to space limitation. 

The model is able to predict the chance of accident for each segment in each 4-h time window. Different spatio temporal predictions for Jan/2020 are shown in the following section based on Logistic regression, different resampling techniques, and with and without resampling. For the sake of comparison, the actual accident occurrence is also shown.  

The figure below demonstrates the general lay out of all the figures.  
![lay_out](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/lay_out.png)  



Figure below shows the actual distribution of accidents over time and space. x-axis denotes the beggining 4-hour time window and y-axis shows the segments; \* blue pixels defines the missing values.  
<img src="https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_testwindow(0)_Actual%20Data.png" width="300">  

The folowing 6 figures are drawn using different prediction models. 
<p float="left">
  <img src="https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+NoR+NoC1_testwindow(0)_Prediction.png" width="250" />
  <img src="https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+RUS+NoC1_testwindow(0)_Prediction.png" width="250" /> 
  <img src="https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+ROS+NoC1_testwindow(0)_Prediction.png" width="250" />
</p>

<p float="left">
  <img src="https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+NoR+KM2_testwindow(0)_Prediction.png" width="250" />
  <img src="https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+RUS+KM2_testwindow(0)_Prediction.png" width="250" /> 
  <img src="https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/spatial_temporal_LR+ROS+KM2_testwindow(0)_Prediction.png" width="250" />
</p>



If the shapefile of the roadway segments are available, the code is capabable of generating heatmap for any intended time windows. For the sake of illustration the figure below, depitcs the aggregated accident rate in Jan/2020. Other figures are also available  in **results_KDD\Figures_not_in_Paper** sub folder in html format.
![LR+RUS+NoC1](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/Map_rate_Actual.png)  


**Report__testwindow(0).html** summerizes **DF_results.pkl** for one test window in a html format. For example, the figure below, illustrates is a screen shot of the **Report__testwindow(0).html** for window test month of January/2020 for all different logistic regression methods. 
![Report](https://github.com/StatResp/KDD_IncidentPrediction/blob/main/results_KDD/Figures_not_in_Paper/Report_LR__testwindow(0).png)
