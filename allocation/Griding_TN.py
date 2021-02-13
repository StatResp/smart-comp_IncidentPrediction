# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 10:19:38 2021

@author: vaziris
"""
'''

'''
from pprint import pprint
import pandas as pd
import geopandas as gpd 
import numpy as np   
import pickle
import json
import pygeoj
import pyproj
import shapely.geometry as sg
from shapely.geometry import Polygon
from shapely.geometry import Point
import swifter
from shapely import ops 
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 100)


#%%




def MyGrouping_Grid(df_inrix,df_incident ,width = 0.1,height = 0.1 ):
    
    '''
    This function grids the state TN and returns the number of segments/groups and highway accidents per grid as well as the boundary and center of each grid
    
        Parameters
    ----------
    df_inrix : DF
        includes the inrix segment for all over TN
    df_incident : DF
        inlcudes the accident records for all over TN

    Returns
    -------
    Grid: GDF
        The Dataframe that incudes the the boundary and center of the grid as well as the the number of segments/groups and highway accidents per grid. The geometry feature is the boundary.
    Grid_center: GDF
        The Dataframe that incudes the the boundary and center of the grid as well as the the number of segments/groups and highway accidents per grid. The geometry feature is the center.
    
    ''' 
    #%%Preparation
    if df_inrix is None: 
        #Reading the Inrix Data
        df_inrix=pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/inrix_grouped.pkl')

    if df_incident is None: 
        #Reading the Incident Data
        df_incident =pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/incident_XDSegID.pkl')
        df_incident=df_incident[df_incident['XDSegID'].notna()]
                

    #Convert to GDF and put the center as the geometry
    df_inrix['line']=df_inrix['geometry']
    df_inrix = gpd.GeoDataFrame(df_inrix, geometry=df_inrix['line'], crs={'init': 'epsg:4326'})
    df_inrix['center']=df_inrix.centroid
    df_inrix = gpd.GeoDataFrame(df_inrix, geometry=df_inrix['center'], crs={'init': 'epsg:4326'})
    
    #Convert to GDF
    df_incident = (gpd.GeoDataFrame(df_incident, geometry=df_incident['geometry'], crs={'init': 'epsg:4326'} ))
    #%%Defining Grid
    xmin,ymin,xmax,ymax =  df_inrix['geometry'].total_bounds
    rows = int(np.ceil((ymax-ymin) /  height))
    cols = int(np.ceil((xmax-xmin) / width))
    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax- height
    polygons = []
    X_id=[]
    Y_id=[]
    for i in range(cols):
       Ytop = YtopOrigin
       Ybottom =YbottomOrigin
       for j in range(rows):
           polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
           X_id.append(i)
           Y_id.append(j)
           Ytop = Ytop - height
           Ybottom = Ybottom - height
       XleftOrigin = XleftOrigin + width
       XrightOrigin = XrightOrigin + width   
    Grid = pd.DataFrame({'geometry':polygons,'X_id':X_id,'Y_id':Y_id }).reset_index().rename(columns={'index':'Grid_ID'})
    Grid['Boundary']=Grid['geometry']
    Grid = (gpd.GeoDataFrame(Grid, geometry=Grid['geometry'], crs={'init': 'epsg:4326'} ))
    Grid['center']=Grid.centroid
    Grid
    #%%Adding Segment
    Grid_Inrix = gpd.sjoin(Grid[['Grid_ID','geometry']], df_inrix[['XDSegID','Miles','geometry']],how="left", op='contains').drop('index_right',axis=1)
    DF_grouped=Grid_Inrix[['Grid_ID','XDSegID','Miles']].groupby('Grid_ID').agg({'XDSegID': ['count'],'Miles': ['sum']}) 
    DF_grouped.columns=['Num_of_Seg','Miles_Seg']
    DF_grouped=DF_grouped.reset_index()
    Grid=pd.merge(Grid,DF_grouped, left_on='Grid_ID', right_on='Grid_ID', how='left' )
    #%%Adding Accident
    Grid_Incident = gpd.sjoin(Grid[['Grid_ID','geometry']], df_incident[['Incident_ID','geometry']],how="left", op='contains').drop('index_right',axis=1)
    DF_grouped=Grid_Incident[['Grid_ID','Incident_ID']].groupby('Grid_ID').agg({'Incident_ID': ['count']}) 
    DF_grouped.columns=['Num_of_Inc']
    DF_grouped=DF_grouped.reset_index()
    Grid=pd.merge(Grid,DF_grouped, left_on='Grid_ID', right_on='Grid_ID', how='left' )
    Grid_center=Grid.copy()
    Grid_center=(gpd.GeoDataFrame(pd.DataFrame(Grid_center), geometry=pd.DataFrame(Grid_center)['center'], crs={'init': 'epsg:4326'} ))
    #%%Graphing
    sns.set()
    Fig1,ax=plt.subplots(2,1,figsize=[20,10])
    Fig1=Grid.plot(column=(np.log(1+Grid['Num_of_Seg'])),ax=ax[0],  legend = True); ax[0].set_title('log of segments per gird')
    Fig2=Grid.plot(column=(np.log(1+Grid['Num_of_Inc'])),ax=ax[1],  legend = True); ax[1].set_title('log of accidents per gird') 
    
    Fig1,ax=plt.subplots(2,1,figsize=[20,10])
    Fig1=Grid.plot(ax=ax[0],  legend = True); ax[0].set_title('Boundary of each grid');ax[0].set_xlim(-91, -81);ax[0].set_ylim(34.5, 37) 
    Fig2=Grid_center.plot(ax=ax[1],  legend = True); ax[1].set_title('Center of each grid');ax[1].set_xlim(-91, -81);ax[1].set_ylim(34.5, 37)      
    
    return Grid,Grid_center





def Distance_Dict_Builder(All_seg_incident,Grid_center):
    '''
    This function builds a dictionary that contains the the distance between the intented segments/groups and Grids

    Parameters
    ----------
    All_seg_incident : DF
        This includes the segments/groups and their geometries, that we want to find their distances to the center of grids .
    Grid_center : GDF
        This is a GDF that for the grids and the geometry is the center of the grid .

    Returns
    -------
    Distant_Dic: Dictionary 
        DESCRIPTION.
    All_seg_incident: GDF
        DESCRIPTION.

    '''

    #%%

    if All_seg_incident is None: 
        df_grouped =pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/grouped/grouped_3.pkl')
        df_=pd.read_pickle('D:/inrix_new/data_main/data/cleaned/Line/Regression_4h_History_G_top20_NoNA.pkl')
        All_seg_incident=df_grouped[df_grouped['Grouping'].isin(df_['XDSegID'])].reset_index().drop('index',axis=1)  #just selecting the groups/segment with at least one accident
             
    #%%    
    All_seg_incident['line']=All_seg_incident['geometry']
    All_seg_incident_4326=gpd.GeoDataFrame(All_seg_incident, geometry=All_seg_incident['line'], crs={'init': 'epsg:4326'} )
    All_seg_incident_4326['center']=All_seg_incident_4326.centroid
    All_seg_incident_4326=gpd.GeoDataFrame(All_seg_incident_4326, geometry=All_seg_incident_4326['center'], crs={'init': 'epsg:4326'} )
    
    
    '''
    def Center_Finder(row):
        #This function finds the the center of the line which connects the beggining and the end
        try:
            Beg=row['geometry'].coords[0]
            End=row['geometry'].coords[-1]
        except: 
            Beg=row['geometry'][0].coords[0]
            End=row['geometry'][-1].coords[-1]    
            
        return Point((Beg[0]+End[0])/2, (Beg[1]+End[1])/2)       
    
    All_seg_incident['line']=All_seg_incident['geometry']
    All_seg_incident['center']=All_seg_incident.apply(lambda row: Center_Finder(row),axis=1)
    All_seg_incident_4326=gpd.GeoDataFrame(All_seg_incident, geometry=All_seg_incident['center'], crs={'init': 'epsg:4326'} )
    '''
 
    All_seg_incident_3310=All_seg_incident_4326.to_crs('EPSG:3310')
    Grid_center_3310=Grid_center.to_crs('EPSG:3310')
    
    #%%
    Distant_Dic={}
    for i,row_i in All_seg_incident_3310.iterrows():
        Distant_Dic[row_i.Grouping]={}
        for j,row_j in Grid_center_3310.iterrows(): 
            Distant_Dic[row_i.Grouping][row_j.Grid_ID]=row_i['geometry'].distance(row_j['geometry'])/1000
    
    
    return Distant_Dic,All_seg_incident




