import configparser
from datetime import datetime
import pytz
from pprint import pprint

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



def FixingTimeZone(metadata):
    """
    
    """
    if metadata['time_zone'].lower()=='local':
        print('Time is set to be local.')
    else:
        pst=pytz.timezone(metadata['time_zone'])
        for TimeLabel in ['start_time_train','end_time_train','start_time_test','end_time_test', 'start_time_predict', 'end_time_predict']:
            metadata[TimeLabel]=pst.localize(metadata[TimeLabel].replace(tzinfo=None))
    return metadata





def create_metadata(config):
    """
    Takes the parsed config file and creates metadata
    @param config: parsed config
    @return: metadata in dictionary form
    """
    metadata = {
        'incident_pickle_address': ConfigSectionMap(config, "filepaths")["incident_pickle"],
        'traffic_pickle_address': ConfigSectionMap(config, "filepaths")["traffic_pickle"],
        'weather_pickle_address': ConfigSectionMap(config, "filepaths")["weather_pickle"],
        'inrix_pickle_address': ConfigSectionMap(config, "filepaths")["inrix_pickle"],
        'groups_pickle_address': ConfigSectionMap(config, "filepaths")["groups_pickle"],
        'regressiondf_pickle_address': ConfigSectionMap(config, "filepaths")["regressiondf_pickle"],
        'window_size': int(ConfigSectionMap(config, "metadata")["window_size"]),
        'unit_name': ConfigSectionMap(config, "metadata")["unit_name"],
        'pred_name_Count': ConfigSectionMap(config, "metadata")["pred_name_count"],
        'pred_name_TF': ConfigSectionMap(config, "metadata")["pred_name_tf"],
        'time_zone': ConfigSectionMap(config, "metadata")["time_zone"],
        'figure_tag':(ConfigSectionMap(config, "metadata")["figure_tag"]=='True'),
        'train_test_type': str(ConfigSectionMap(config, "metadata")["train_test_type"]),
        'train_test_split': float(ConfigSectionMap(config, "metadata")["train_test_split"]),
        'train_verification_split': float(ConfigSectionMap(config, "metadata")["train_verifcation_split"]),
        'start_time_train': datetime.strptime(ConfigSectionMap(config, "metadata")["start_time_train"], '%d-%m-%Y %H:%M:%S'),
        'end_time_train': datetime.strptime(ConfigSectionMap(config, "metadata")["end_time_train"],     '%d-%m-%Y %H:%M:%S'),
        'start_time_test': datetime.strptime(ConfigSectionMap(config, "metadata")["start_time_test"], '%d-%m-%Y %H:%M:%S'),
        'end_time_test': datetime.strptime(ConfigSectionMap(config, "metadata")["end_time_test"],     '%d-%m-%Y %H:%M:%S'),
        'start_time_predict': datetime.strptime(ConfigSectionMap(config, "metadata")["start_time_predict"],   '%d-%m-%Y %H:%M:%S'),
        'end_time_predict': datetime.strptime(ConfigSectionMap(config, "metadata")["end_time_predict"],       '%d-%m-%Y %H:%M:%S')}   
    metadata=FixingTimeZone(metadata)
    
    
    #Regeression Parameters
    #1) Parameters from DFs
    metadata['features']=[]
    f       = ConfigSectionMap(config, "regressionParams")
    for Name in ['features_temporal', 'features_indident', 'features_weather', 'features_traffic','features_static']:
        try: 
            metadata[Name] = [x.strip() for x in f[Name].split(',')]
            metadata['features'].extend(metadata[Name])
        except:   
            metadata[Name] = None
    #3) Categorical Params
    try:    
        metadata['cat_features'] = [x.strip() for x in f["categorical_features"]  .split(',')]
    except:
        metadata['cat_features']=None        
    #4) train test split for the training of reg model
    #metadata['train_test_split']    = float(f["train_test_split"])



    #f_stat  = ConfigSectionMap(config, "clusterParams")["static_features"]
    #metadata['Cluster_type']  = ConfigSectionMap(config, "clusterParams")["cluster_type"]
    #metadata['Cluster_numbers']  = int(ConfigSectionMap(config, "clusterParams")["cluster_numbers"])

    f_clusters= ConfigSectionMap(config, "clusterParams")
    metadata['cluster_type']  = [x.strip() for x in f_clusters['cluster_type'].split(',')]  
    metadata['cluster_number']  = [int(x) for x in f_clusters['cluster_number'].split(',')]  


    f_models= ConfigSectionMap(config, "regressionModels")["model_type"]
    metadata['model_type'] = [x.strip() for x in f_models.split(',')]
    f_models_sampling= ConfigSectionMap(config, "regressionModels")["resampling_type"]
    metadata['resampling_type'] = [x.strip() for x in f_models_sampling.split(',')]    
    
    return metadata


def read_config(Address):
    """
    reads the config file and returns metadata dictionary
    @return:
    """
    # READ CONFIG
    config = configparser.ConfigParser()
    config.read(Address)
    metadata = create_metadata(config)
    return metadata






