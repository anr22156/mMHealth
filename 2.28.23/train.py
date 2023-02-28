#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing standard libraries
import pandas as pd
import itertools
import scipy
import pymongo
import argparse
import matplotlib
matplotlib.use('tkagg')

import config
from container import Container
from DataSource import DataSource

import preprocessing_methods1

import joblib
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split


# In[ ]:


argparse = argparse.ArgumentParser()
argparse.add_argument("-ds", "-- data-source", required=True, dest="data_source", 
                      type=str, default=DataSource.COLLECTION1.value)
args = vars(argparse.parse_args())


# In[ ]:


container = Container()
dataframe = container.prepared_data_provider().get(args['data_source'])


# In[ ]:


# Split dataframe into lists
PPG_data = dataframe['PPG'].tolist()
ECG_data = dataframe['ECG'].tolist()
ABP_data = dataframe['BP'].tolist()

#dataframe.to_csv('df_test1.csv')

# In[ ]:


# Run preprocessing methods on the data
ECG_features = preprocessing_methods1.ECGPreprocessing(ECG_data)
PPG_features, PPG_norm = preprocessing_methods1.PPGPreprocessing(PPG_data)
ABP_features = preprocessing_methods1.BPPreprocessing(ABP_data)

# Run preprocessing methods for combining the data
df_init = preprocessing_methods1.combine_data(ECG_features, PPG_features, ABP_features)
df = preprocessing_methods1.features(df_init,PPG_norm)


# In[ ]:


main_data, test_data = train_test_split(df, test_size=0.20, random_state=42, shuffle=True)
test_data.to_csv(config.model['test_data_file'])


# In[ ]:


output = ['sys_val','dias_val']
y = main_data[output]
X = main_data.drop(columns=output) 


# In[ ]:


# Training/Testing the ML Model - Random Forest
# Scaling the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# In[ ]:


model_builder = container.model_builder()
regressor = model_builder.build()
regressor.fit(X, y)


# In[ ]:


joblib.dump(regressor, config.model['model_file_name'])
joblib.dump(scaler, config.model['sklearn_scaler_file_name'])

