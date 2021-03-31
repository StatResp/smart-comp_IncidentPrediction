import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pygeoj
import pyproj
from allocation.Griding_TN import MyGrouping_Grid, Distance_Dict_Builder

from allocation.pmedianAllocator import pmedianAllocator


df_inrix=None
df_incident=None
Grid, Grid_center=MyGrouping_Grid(df_inrix,df_incident ,width = 0.1,height = 0.1 )



#Just for test
Grid_3310 =Grid_center.to_crs('EPSG:3310')
Grid_3310['center_3310']=Grid_3310.centroid
Grid_3310['area']=Grid.to_crs('EPSG:3310').area/1e6

All_seg_incident=None
Distant_Dic,All_seg_incident=Distance_Dict_Builder(All_seg_incident,Grid_center)


allocator = pmedianAllocator()


# inputs
num_resources = 5
possible_facility_locations = set(Grid_center['Grid_ID'])
demand_nodes = set(All_seg_incident['Grouping'])

weights_dict = dict()
for n_l in demand_nodes:
    weights_dict[n_l] = float(n_l) / 10000000.0

allocation = allocator.solve(number_of_resources_to_place=num_resources,
                             possible_facility_locations= possible_facility_locations,
                             demand_nodes=demand_nodes,
                             distance_dict=Distant_Dic,
                             demand_weights=weights_dict,
                             score_type='penalty',
                             alpha=1.0)

print(allocation)
#fig=Grid.plot()
All_seg_incident_4326=gpd.GeoDataFrame(All_seg_incident, geometry=All_seg_incident['line'], crs={'init': 'epsg:4326'} )
fig=All_seg_incident_4326.plot(All_seg_incident_4326['Grouping'])
Grid_center[Grid_center['Grid_ID'].isin(allocation)].plot(ax=fig)

All_seg_incident_4326=gpd.GeoDataFrame(All_seg_incident, geometry=All_seg_incident['line'], crs={'init': 'epsg:4326'} )
All_seg_incident_4326.plot(All_seg_incident_4326['Grouping'])



print('done!')







