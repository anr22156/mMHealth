#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing standard libraries
import pandas as pd
import numpy as np
import itertools
import scipy
import pymongo
import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import config
from container import Container
from model.DataSource import DataSource

import joblib
from sklearn.model_selection import train_test_split


# In[ ]:


argparse = argparse.ArgumentParser()
argparse.add_argument("-ds", "-- data-source", required=True, dest="data_source", 
                      type=str, help="What datasource to use: COLLECTION1 or COLLECTION2")
args = vars(argparse.parse_args())


# In[ ]:


container = Container()
dataframe = container.prepared_data_provider().get(args['data_source'])


# In[ ]:


# Split dataframe into lists
PPG = dataframe['PPG'].tolist()
ECG = dataframe['ECG'].tolist()


# In[ ]:


# Run preprocessing methods on the data
new_ECG_features = preprocessing_methods1.ECGPreprocessing(new_ECG)
new_PPG_features, new_PPG_norm = preprocessing_methods1.PPGPreprocessing(new_PPG)

# Run preprocessing methods for combining the data
new_df_init = preprocessing_methods1.combine_data1(new_ECG_features, new_PPG_features, 60)
new_df = preprocessing_methods1.features(new_df_init, new_PPG_norm)
new_df = new_df.replace([np.inf, -np.inf], np.nan).dropna()


# In[ ]:


scaler = joblib.load(config.model['sklearn_scaler_file_name'])
regressor = joblib.load(config.model['model_file_name'])


# In[ ]:


y_pred = regressor.predict(scaler.transform(new_df))

pred_sys = [item[0] for item in y_pred]
pred_dias = [item[1] for item in y_pred]


# In[ ]:


fig,[ax1,ax2] = plt.subplots(2,1,figsize=(12,5),sharex=True)
ax1.plot(pred_dias,'r-')
ax1.plot(y['dias_val'],'b--')
ax1.set_title('Diastolic BP')
ax1.legend(['pred','test'])
ax1.set_ylim([40,160])
ax2.plot(pred_sys,'r-')
ax2.plot(y['sys_val'],'b--')
ax2.legend(['pred','test'])
ax2.set_title('Systolic BP')
ax2.set_ylim([80,200])
plt.show()

