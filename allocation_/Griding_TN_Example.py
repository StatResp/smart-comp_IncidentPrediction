# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:43:32 2021

@author: vaziris
"""
#This is example how to run the Griding_TN and its function:

import pandas as pd
import geopandas as gpd 
import matplotlib.pyplot as plt
import pygeoj
import pyproj
from allocation.Griding_TN import MyGrouping_Grid, Distance_Dict_Builder


df_inrix=None
df_incident=None
Grid, Grid_center=MyGrouping_Grid(df_inrix,df_incident ,width = 0.1,height = 0.1 )



#Just for test
Grid_3310 =Grid_center.to_crs('EPSG:3310')
Grid_3310['center_3310']=Grid_3310.centroid
Grid_3310['area']=Grid.to_crs('EPSG:3310').area/1e6







All_seg_incident=None
Distant_Dic,All_seg_incident=Distance_Dict_Builder(All_seg_incident,Grid_center)
Distant_Dic



#Sanity Check
#If you want to draw the center of the line and the line: 
i=339; j=i+1 #range of the segmenet/group you want to draw
    

All_seg_incident_line=pd.DataFrame(All_seg_incident).copy()
All_seg_incident_center=pd.DataFrame(All_seg_incident).copy()
All_seg_incident_line=gpd.GeoDataFrame(All_seg_incident_line, geometry=All_seg_incident_line['line'], crs={'init': 'epsg:4326'} )
All_seg_incident_center=gpd.GeoDataFrame(All_seg_incident_center, geometry=All_seg_incident_center['center'], crs={'init': 'epsg:4326'}) 
Fig1,ax=plt.subplots(figsize=[10,10])
Fig1=All_seg_incident_line.iloc[i:j].plot(All_seg_incident_center.iloc[i:j]['Grouping'],ax=ax)
Fig1=All_seg_incident_center.iloc[i:j].plot(All_seg_incident_center.iloc[i:j]['Grouping'],ax=Fig1);ax.set_title('location of the line and its center on the map; EPSG:4326');



#if you want to see the calculated distance and the distance on the map run the following code
XDseg=3012476
Grid_ID=1000
Fig1,ax=plt.subplots(2,1,figsize=[20,10])
Fig1=Grid[Grid['Grid_ID']==Grid_ID].plot(ax=ax[0],  legend = True);                         ax[0].set_title('EPSG:4326');
Grid_center[Grid_center['Grid_ID']==Grid_ID].plot(ax=Fig1, color='black')
All_seg_incident_line[All_seg_incident_line['Grouping']==XDseg].plot(ax=Fig1)
All_seg_incident_center[All_seg_incident_center['Grouping']==XDseg].plot(ax=Fig1, color='black')

Grid_3310=Grid.to_crs('EPSG:3310')
Grid_center_3310=Grid_center.to_crs('EPSG:3310')
All_seg_incident_line_3310=All_seg_incident_line.to_crs('EPSG:3310')
All_seg_incident_center_3310=All_seg_incident_center.to_crs('EPSG:3310')
new_df_3310 = Grid_center_3310[Grid_center_3310['Grid_ID']==Grid_ID].copy()
new_df_3310['geometry'] = new_df_3310['geometry'].buffer(Distant_Dic[XDseg][Grid_ID]*1000)
    
Fig2=Grid_3310[Grid_3310['Grid_ID']==Grid_ID].plot(ax=ax[1],  legend = True);                         ax[1].set_title('EPSG:3310');
Grid_center_3310[Grid_center_3310['Grid_ID']==Grid_ID].plot(ax=Fig2, color='black')
All_seg_incident_line_3310[All_seg_incident_line_3310['Grouping']==XDseg].plot(ax=Fig2)
All_seg_incident_center_3310[All_seg_incident_center_3310['Grouping']==XDseg].plot(ax=Fig2, color='black')
new_df_3310.plot(ax=Fig2, color='red', alpha=0.1)

new_df_4326=new_df_3310.to_crs('EPSG:4326')
new_df_4326.plot(ax=Fig1, color='red', alpha=0.1)


print('THe calculated distance is: ', Distant_Dic[XDseg][Grid_ID], ' km')

