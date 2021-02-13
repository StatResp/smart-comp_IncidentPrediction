"""
@Author - Sayyed Mohsen Vazirizade
Plotting methods for forecasting models
"""

#packages______________________________________________________________
from scipy.interpolate import UnivariateSpline
import numpy as np
import os
import pandas as pd
import pyproj
import math
import matplotlib.pyplot as plt
import shapely.geometry as sg
import folium
import geopandas as gpd
import random
import pprint
import seaborn as sns
from datetime import timedelta
from math import ceil, floor
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from copy import deepcopy
import geopandas as gpd
import folium



def MatrixMaker(df,metadata,NROWS, NCOLS,SeedNum,COLNAME):
    np.random.seed(seed=SeedNum)        
    HeatmapMatrix=df.pivot(index=metadata['unit_name'], columns='time_local')[COLNAME]
    HeatmapMatrix=HeatmapMatrix.sample(min(NROWS,len(HeatmapMatrix)),replace=False).sort_values(by=metadata['unit_name'])
    np.random.seed(seed=SeedNum) 
    random.seed(SeedNum)
    ListofCols=sorted(random.sample(list(np.arange(0,    len(HeatmapMatrix.columns))), min(NCOLS,len(HeatmapMatrix.columns))     ))
    HeatmapMatrix=HeatmapMatrix[HeatmapMatrix.columns[ListofCols]]
    try: 
        HeatmapMatrix.columns=HeatmapMatrix.columns.strftime('%Y-%m-%d %H')        #('%Y-%m-%d %H:%M:%S')
    except:
        HeatmapMatrix.columns=pd.DataFrame(HeatmapMatrix.columns).time_local.apply(lambda x: x.strftime('%Y-%m-%d %H')) 
        #HeatmapMatrix.columns=pd.DataFrame(HeatmapMatrix.columns).time.apply(lambda x: x.strftime('%Y-%m-%d %H'))        #('%Y-%m-%d %H:%M:%S') 
    HeatmapMatrix.index=[int(i) for i in HeatmapMatrix.index] 
    return HeatmapMatrix
    



def Heatmap(df,metadata,model_i,COLNAME=None,maxrange=1,NROWS=100, NCOLS=100,SeedNum=0 ):
    HeatmapMatrix=MatrixMaker(df,metadata,NROWS=NROWS, NCOLS=NCOLS,SeedNum=SeedNum,COLNAME=COLNAME)
    plt.figure(figsize=(10,10))
    plt.suptitle(model_i, fontsize=16)
    plt.tight_layout()
    Subplot_size=10
    ax1 = plt.subplot2grid((Subplot_size,Subplot_size), (1,2),  colspan=Subplot_size-3, rowspan=Subplot_size-1) # middle
    sns.heatmap(data=HeatmapMatrix,vmin=0, vmax=maxrange, cmap='rocket',facecolor='y', cbar=False, ax=ax1)  #sns.cm.rocket viridis
    ax1.set_facecolor('blue')

    ax2 = plt.subplot2grid((Subplot_size,Subplot_size), (0,2),  colspan=Subplot_size-3, rowspan=1)            # top
    sns.heatmap(data=HeatmapMatrix.mean(axis=0).values.reshape(1,HeatmapMatrix.shape[1]), vmin=0, vmax=maxrange, cmap='rocket',facecolor='y', cbar=False, ax=ax2)
    ax2.set_facecolor('blue'); ax2.set_ylabel(''); ax2.set_xlabel('');ax2.set(xticklabels=[]);ax2.set(yticklabels=[])  

    ax3 = plt.subplot2grid((Subplot_size,Subplot_size), (1,Subplot_size-1),  colspan=1, rowspan=Subplot_size-1)   #right
    sns.heatmap(data=HeatmapMatrix.mean(axis=1).values.reshape(HeatmapMatrix.shape[0],1),vmin=0, vmax=maxrange, cmap='rocket',facecolor='y', cbar=True, ax=ax3)
    ax3.set_facecolor('blue'); ax3.set_ylabel(''); ax3.set_xlabel('');ax3.set(xticklabels=[]);ax3.set(yticklabels=[])  

    HeatmapMatrix=pd.DataFrame({'Mean': [HeatmapMatrix.mean(axis=0).mean()]})
    ax5 = plt.subplot2grid((Subplot_size,Subplot_size), (0, Subplot_size-1),  colspan=1, rowspan=1)    #top right
    sns.heatmap(data=HeatmapMatrix,vmin=0, vmax=maxrange, cmap='rocket',facecolor='y', cbar=False, ax=ax5)
    ax5.set_facecolor('blue'); ax5.set_ylabel(''); ax5.set_xlabel('');ax5.set(xticklabels=[]);ax5.set(yticklabels=[])   

    HeatmapMatrix=MatrixMaker(df,metadata,NROWS=NROWS, NCOLS=NCOLS,SeedNum=SeedNum,COLNAME='cluster_label') #left
    ax4 = plt.subplot2grid((Subplot_size,Subplot_size), (1,0),  colspan=1, rowspan=Subplot_size-1)   
    sns.heatmap(data=HeatmapMatrix.mean(axis=1).values.reshape(HeatmapMatrix.shape[0],1),cmap='viridis',facecolor='y', cbar=False, ax=ax4)
    ax4.set_facecolor('blue'); ax4.set_ylabel(''); ax4.set_xlabel('Clusters');ax4.set(xticklabels=[]);ax4.set(yticklabels=[])  

    
    
def Heatmap_cluster(df,metadata,model_i,maxrange=10,NROWS=100, NCOLS=100,SeedNum=0 ):
    df_predict=deepcopy(df)
    for i in df_predict['cluster_label'].unique().tolist():
        df_predict.loc[df_predict["cluster_label"]==i,[metadata['pred_name_TF']]]=i
    Heatmap(df_predict,metadata,model_i,COLNAME=metadata['pred_name_TF'],vmin=0,maxrange=maxrange,NROWS=NROWS, NCOLS=NCOLS,SeedNum=SeedNum) 
    





def Heatmap_on_Map_TimeRange(DF, time_range, metadata, Feature_List=None, Name=' ',number=5000):
    #DF=learn_results[model_i]['df_predict']
    #number=5000
    
    # Heatmap_on_Map(learn_results[m]['df_predict'],learn_results[m]['df_predict']['time'].iloc[0] ) 
    
    
    metadata['unit_name']
    Predict_df=DF[DF['time_local'].isin(time_range)].copy()
    Predict_df=Predict_df.rename(columns={'XDSegID':'MyGrouping_3'})
    
    Predict_df=Predict_df[['MyGrouping_3']+Feature_List].groupby('MyGrouping_3').mean().reset_index()
    
    
    #maping my MyGrouping_3 to actual XDSegID
    #FRC0 = pd.read_pickle('D:/inrix/inrix/other/inrix_FRC0_etrims_3D_curvature_df.pk')
    #FRC0 = pd.read_pickle('sample_data/data_cleaned_inrix_grouped.pkl')
    FRC0 =pd.read_pickle(metadata['inrix_pickle_address'])
    #Predict_df=pd.merge(Predict_df[['time','time_local',metadata['pred_name_TF'],'cluster_label','predicted','MyGrouping_3']], FRC0[['XDSegID','MyGrouping_3']], left_on='MyGrouping_3', right_on='MyGrouping_3', how='right')
    Predict_df=pd.merge(Predict_df, FRC0[['XDSegID','MyGrouping_3','geometry' ]], left_on='MyGrouping_3', right_on='MyGrouping_3', how='inner')
    
    #maping XDSegID to geometry
    #static_df = pd.read_pickle(metadata['static_pickle_address'])
    #static_df=pd.read_pickle("D:/inrix/inrix/other/inrix_pd_2D.pk")
    #Predict_df=pd.merge(Predict_df, static_df[['geometry','XDSegID']], left_on='XDSegID', right_on='XDSegID', how='left')
    
    IDs=range(0,min(number,len(Predict_df)))
    Predict_df['geometry']= [i.buffer(0.025) for i in Predict_df['geometry']]  
    
    plot_area= gpd.GeoDataFrame(data=Predict_df.iloc[IDs][[metadata['unit_name'],'geometry']])
    plot_area = plot_area.to_json()
    Map = folium.Map(location = [35.151, -86.852], opacity=0.5, tiles='cartodbdark_matter', zoom_start = 8)   #'Stamen Toner'
    #title_html = '''     <h3 align="center" style="font-size:16px"><b>{}</b></h3>  '''.format(time_range[0].strftime('%Y-%m-%d %H:%M:%S %Z'))
    Title=[i.strftime('%Y-%m-%d %H:%M:%S %Z') for i in [time_range[0],time_range[-1] ]]
    Title=[Title[0]+' to '+Title[1] + Name]
    print(Title[0])
    title_html = '''     <h3 align="center" style="font-size:16px"><b>{}</b></h3>  '''.format(Title[0])
    Map.get_root().html.add_child(folium.Element(title_html))
    #Min=min(DF['predicted'].min(),DF[metadata['pred_name_TF']].min())
    #Max=max(DF['predicted'].max(),DF[metadata['pred_name_TF']].max())
    Range=[0, 0.1, 0.25, 0.5 , 0.75,1]
    
    Key_on_tag='feature.properties.'+metadata['unit_name']
    #'feature.properties.unit_segment_id'
    for Feature in Feature_List:
            folium.Choropleth(
                geo_data=plot_area,
                name=Feature,#+time_point.strftime('%Y-%m-%d %H'),
                data=Predict_df,
                columns=[metadata['unit_name'],Feature],
                key_on=Key_on_tag,
                fill_color = 'Reds', line_color = 'white', fill_opacity = 1,line_opacity = 0.5 ,line_weight=0.001, #RdYlBu_r   #I coulndt find rocket cmap for folium!
                #threshold_scale=Range,
                #threshold_scale=[i/DF[metadata['pred_name_TF']].max() for i in Range],
                legend_name = Feature).add_to(Map)         
            
    folium.LayerControl().add_to(Map)
    #    Map.save("Map_rate.html")      
    return(Map)  





def Graph_Metric(DF_summary,Title):
    #print(DF)

    import matplotlib
    from matplotlib import pyplot
    import seaborn as sns
    
    #plt.figure(figsize=[20,10])
    fig, axis = pyplot.subplots(4,1,figsize=(15,15))
    sns.set_palette("tab10")
    DF_summary_=DF_summary.set_index('time_local')
    metric_list=['accuracy','recall','precsion','F1']
    for i in range(4):
        DF=DF_summary_[[j for j in DF_summary_.columns if j.split('_')[-1]==metric_list[i]]]
        ax=axis[i]
        DF.plot(linewidth=1,ax=ax) 
        ax.set_xlabel('')
        ax.set_ylabel(metric_list[i])
        #ax.set_title(metric_list[i])
        ax.set_xticks([],minor=False)   #position

        for i,Value in enumerate(DF.mean(axis=0)):
            ax.axhline(Value, ls='--',linewidth=2,color=sns.color_palette("tab10")[i])
    ax.set_xlabel('Time (Local)')
    ax.set_xticks(DF.index[::6],minor=False)   #position
    ax.set_xticklabels(DF.index[::6], rotation=90,minor=False) 	  #label rotation, etc. 
    ax.set_xticks([],minor=True)   #position
    axis[0].set_title(Title)    
    plt.show()



'''

# ('feature.properties.'+metadata['unit_name'])   
def Heatmap_on_Map(DF, time_point, metadata, number=5000):
    #DF=learn_results[model_i]['df_predict']
    #number=5000
    
    # Heatmap_on_Map(learn_results[m]['df_predict'],learn_results[m]['df_predict']['time'].iloc[0] ) 
    metadata['unit_name']
    Predict_df=DF[DF['time_local']==time_point].copy()
    Predict_df=Predict_df.rename(columns={'XDSegID':'MyGrouping_3'})
    
    
    #maping my MyGrouping_3 to actual XDSegID
    #FRC0 = pd.read_pickle('D:/inrix/inrix/other/inrix_FRC0_etrims_3D_curvature_df.pk')
    FRC0 = pd.read_pickle('sample_data/data_cleaned_inrix_grouped.pkl')
    #Predict_df=pd.merge(Predict_df[['time','time_local','count','cluster_label','predicted','MyGrouping_3']], FRC0[['XDSegID','MyGrouping_3']], left_on='MyGrouping_3', right_on='MyGrouping_3', how='right')
    Predict_df=pd.merge(Predict_df[['time','time_local','count','cluster_label','predicted','MyGrouping_3']], FRC0[['XDSegID','MyGrouping_3','geometry' ]], left_on='MyGrouping_3', right_on='MyGrouping_3', how='inner')
    
    #maping XDSegID to geometry
    #static_df = pd.read_pickle(metadata['static_pickle_address'])
    #static_df=pd.read_pickle("D:/inrix/inrix/other/inrix_pd_2D.pk")
    #Predict_df=pd.merge(Predict_df, static_df[['geometry','XDSegID']], left_on='XDSegID', right_on='XDSegID', how='left')
    
    IDs=range(0,min(number,len(Predict_df)))
    Predict_df['geometry']= [i.buffer(0.01) for i in Predict_df['geometry']]  
    
    plot_area= gpd.GeoDataFrame(data=Predict_df.iloc[IDs][[metadata['unit_name'],'geometry',metadata['pred_name'],'predicted']])
    plot_area = plot_area.to_json()
    Map = folium.Map(location = [36.151, -86.852], opacity=0.5, tiles='Stamen Toner', zoom_start = 12)
    title_html = '     <h3 align="center" style="font-size:16px"><b>{}</b></h3>  '.format(time_point.strftime('%Y-%m-%d %H:%M:%S %Z'))
    Map.get_root().html.add_child(folium.Element(title_html))
    #Min=min(DF['predicted'].min(),DF[metadata['pred_name']].min())
    #Max=max(DF['predicted'].max(),DF[metadata['pred_name']].max())
    Range=[0, 0.1, 0.25, 0.5 , 0.75,1]
    
    Key_on_tag='feature.properties.'+metadata['unit_name']
    #'feature.properties.unit_segment_id'
    if "count" in DF.columns:
        folium.Choropleth(
            geo_data=plot_area,
            name=metadata['pred_name']+time_point.strftime('%Y-%m-%d %H'),
            data=Predict_df,
            columns=[metadata['unit_name'],metadata['pred_name']],
            key_on=Key_on_tag,
            fill_color = 'Reds', line_color = 'black', fill_opacity = 1,line_opacity = 1,line_weight=0.001, #RdYlBu_r   #I coulndt find rocket cmap for folium!
            threshold_scale=Range,
            #threshold_scale=[i/DF[metadata['pred_name']].max() for i in Range],
            legend_name = 'Rate Count').add_to(Map)         
        
    else:
        print('No count column found!')
    if "predicted" in DF.columns:
        folium.Choropleth(
            geo_data=plot_area,
            name='Predict'+time_point.strftime('%Y-%m-%d %H'),
            data=Predict_df,
            columns=[metadata['unit_name'],'predicted'],
            key_on=Key_on_tag,
            fill_color = 'Reds', line_color = 'black', fill_opacity = 1,line_opacity = 1,line_weight=0.001,
            threshold_scale=Range,
            legend_name = 'Rate Prediction').add_to(Map) 
    else:
        print('No count column found!')
    folium.LayerControl().add_to(Map)
    #    Map.save("Map_rate.html")      
    return(Map)  



def Clusters_on_Map(DF, time_point, metadata, number=5000):
    # Heatmap_on_Map(learn_results[m]['df_predict'],learn_results[m]['df_predict']['time'].iloc[0] ) 
    metadata['unit_name']
    Predict_df=DF[DF['time_local']==time_point].copy()
    Predict_df=Predict_df.rename(columns={'XDSegID':'MyGrouping_3'})
    
    
    #maping my MyGrouping_3 to actual XDSegID
    #FRC0 = pd.read_pickle('D:/inrix/inrix/other/inrix_FRC0_etrims_3D_curvature_df.pk')
    FRC0 = pd.read_pickle('sample_data/data_cleaned_inrix_grouped.pkl')
    #Predict_df=pd.merge(Predict_df[['time','time_local','count','cluster_label','predicted','MyGrouping_3']], FRC0[['XDSegID','MyGrouping_3']], left_on='MyGrouping_3', right_on='MyGrouping_3', how='right')
    Predict_df=pd.merge(Predict_df[['time','time_local','count','cluster_label','predicted','MyGrouping_3']], FRC0[['XDSegID','MyGrouping_3','geometry' ]], left_on='MyGrouping_3', right_on='MyGrouping_3', how='inner')
    
    #maping XDSegID to geometry
    #static_df = pd.read_pickle(metadata['static_pickle_address'])
    #static_df=pd.read_pickle("D:/inrix/inrix/other/inrix_pd_2D.pk")
    #Predict_df=pd.merge(Predict_df, static_df[['geometry','XDSegID']], left_on='XDSegID', right_on='XDSegID', how='left')
    
    IDs=range(0,min(number,len(Predict_df)))
    Predict_df['geometry']= [i.buffer(0.01) for i in Predict_df['geometry']]  
    
    plot_area= gpd.GeoDataFrame(data=Predict_df.iloc[IDs][[metadata['unit_name'],'geometry',metadata['pred_name'],'predicted']])
    plot_area = plot_area.to_json()
    Map = folium.Map(location = [36.151, -86.852], opacity=0.5, tiles='Stamen Toner', zoom_start = 12)
    title_html = '     <h3 align="center" style="font-size:16px"><b>{}</b></h3>  '.format(time_point.strftime('%Y-%m-%d %H:%M:%S %Z'))
    Map.get_root().html.add_child(folium.Element(title_html))
    #Min=min(DF['predicted'].min(),DF[metadata['pred_name']].min())
    #Max=max(DF['predicted'].max(),DF[metadata['pred_name']].max())

    Key_on_tag='feature.properties.'+metadata['unit_name']
    #'feature.properties.unit_segment_id'
    if "cluster_label" in DF.columns:
        folium.Choropleth(
            geo_data=plot_area,
            name=metadata['pred_name']+time_point.strftime('%Y-%m-%d %H'),
            data=Predict_df,
            columns=[metadata['unit_name'],metadata['pred_name']],
            key_on=Key_on_tag,
            fill_color ='RdYlGn',line_color = 'black', fill_opacity = 1,line_opacity = 1,line_weight=0.001, #RdYlBu_r   #I coulndt find rocket cmap for folium!
            #threshold_scale=Range,
            legend_name = 'Rate Count').add_to(Map)    
    else:
        print('No count column found!')
    folium.LayerControl().add_to(Map)
    #    Map.save("Map_rate.html")      
    return(Map)  




'''












'''
    
#drawing segments in an order range on the map
#IDs=range(0,len(df))
Predict=learn_results[m]['df_predict']
Predict=Predict[Predict['time']==learn_results[m]['df_predict']['time'].iloc[0]]

static_df = pd.read_pickle(metadata['static_pickle_address'])

DF=pd.merge(Predict, static_df[['geometry','seg_id']], left_on='unit_segment_id', right_on='seg_id', how='inner')

#1
colors = ['red','blue','gray','darkred','lightred','orange','beige','green','darkgreen','lightgreen','darkblue','lightblue','purple','darkpurple','pink','cadetblue',    'lightgray']
#2
def get_color(props):
  Curve_Turn_Min = props['Curve_Turn_Min']
  if Curve_Turn_Min< 1:
    return 'red'
  elif Curve_Turn_Min <100 :
    return 'yellow'
  else:
    return 'blue'
#3
#import folium.colormap as cm
import branca.colormap as cm
linearcol = cm.LinearColormap(['green','yellow','red'], vmin=0., vmax=6)
linearcol(9)

IDs=range(0,300)
df1= gpd.GeoDataFrame(data=DF.iloc[IDs][['unit_segment_id','geometry','predicted',metadata['pred_name']]])
plot_data = df1.to_json()
m1 = folium.Map(location=[36.151, -86.852], tiles='Stamen Toner',zoom_start=14)
#style_function = lambda x: {"color": linearcol(x['properties']), "weight": 8} 
#style_function = lambda x: {"color": random.choice(colors),"weight":5}
style_function = lambda x: {"color": linearcol(x['properties'].predicted),"weight":5}
folium.GeoJson(plot_data, style_function=style_function, name='predicted').add_to(m1)
folium.LayerControl().add_to(m1)
m1  
m1.save("Map.html")      
    
    

    








    
            

NROWS=30
NCOLS=30
SeedNum=0 
maxrange=6
df=learn_results[m]['df_predict']
np.random.seed(seed=SeedNum)        
HeatmapMatrix=df.pivot(index=metadata['unit_name'], columns='time')[metadata['pred_name']]
HeatmapMatrix=HeatmapMatrix.sample(min(NROWS,len(HeatmapMatrix)),replace=False).sort_values(by=metadata['unit_name'])
ListofCols=sorted(random.sample(list(np.arange(0,    len(HeatmapMatrix.columns))), min(NCOLS,len(HeatmapMatrix.columns))     ))
HeatmapMatrix=HeatmapMatrix[HeatmapMatrix.columns[ListofCols]]
HeatmapMatrix.columns=HeatmapMatrix.columns.strftime('%Y-%m-%d %H')        #('%Y-%m-%d %H:%M:%S')
HeatmapMatrix.index=[int(i) for i in HeatmapMatrix.index]
plt.figure(figsize=(10,10))
g=sns.heatmap(data=HeatmapMatrix,vmin=0, vmax=maxrange, cmap='rocket')  #sns.cm.rocket 



def HeatmapTrain(df_train,metadata):    
    windows = ceil(int((metadata['end_time_train'] - metadata['time_train']).total_seconds()) / metadata['window_size'])
    temp_start = metadata['start_time_test']
    Alltimes=[]
    for i in range(windows):
        Alltimes.append(temp_start)
        temp_start += timedelta(seconds=metadata['window_size'])
        
    unit_name=metadata['unit_name']
    DrawingDF=pd.DataFrame()
    DrawingDF[unit_name]=df_train[unit_name]
    DrawingDF['Time']= np.repeat(Alltimes, len(metadata['units']))
    DrawingDF[metadata['pred_name']]=df_train[metadata['pred_name']]
    DrawingDF=DrawingDF.sort_values(by=['Time',unit_name])
    
    HeatmapMattest=DrawingDF.pivot(index=unit_name, columns='Time')[metadata['pred_name']]
    plt.figure(figsize=(10,10))
    g=sns.heatmap(data=HeatmapMattest)
        
    
def HeatmapSample(df_samples,metadata):    
    windows = ceil(int((metadata['end_time_test'] - metadata['start_time_test']).total_seconds()) / metadata['window_size'])
    temp_start = metadata['start_time_test']
    Alltimes=[]
    for i in range(windows):
        Alltimes.append(temp_start)
        temp_start += timedelta(seconds=metadata['window_size'])
        
    unit_name=metadata['unit_name']
    DrawingDF=pd.DataFrame()
    DrawingDF[unit_name]=df_samples[unit_name]
    DrawingDF['Time']= np.repeat(Alltimes, len(metadata['units']))
    DrawingDF['sample']=df_samples['sample']
    DrawingDF=DrawingDF.sort_values(by=['Time',unit_name])
    
    HeatmapMattest=DrawingDF.pivot(index=unit_name, columns='Time')['sample']
    plt.figure(figsize=(10,10))
    g=sns.heatmap(data=HeatmapMattest)
            
'''