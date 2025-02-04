import pandas as pd
import itertools
import scipy
import argparse

import config
from container import Container
from DataSource import DataSource

import preprocessing_methods1


# Function that completes all of the preprocessing and feature detection methods for a data collection
# Returns a dataframe with all of the features that have been extracted

def process_docs(collection_name):
    
    container = Container()
    dataframe = container.prepared_data_provider().get(collection_name)

    PPG_data = dataframe['PPG'].tolist()
    ECG_data = dataframe['ECG'].tolist()
    ABP_data = dataframe['BP'].tolist()
    
    ECG_features = preprocessing_methods1.ECGPreprocessing(ECG_data)
    PPG_features, PPG_norm = preprocessing_methods1.PPGPreprocessing(PPG_data)
    ABP_features = preprocessing_methods1.BPPreprocessing(ABP_data)

    df_init = preprocessing_methods1.combine_data(ECG_features, PPG_features, ABP_features)
    df = preprocessing_methods1.features(df_init,PPG_norm)
    
    return df
