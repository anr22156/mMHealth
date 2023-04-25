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
from DataSource import DataSource
import preprocessing_methods1

import joblib
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

# In[ ]:


argparse = argparse.ArgumentParser()
argparse.add_argument("-ds", "-- data-source", required=True, dest="data_source", type=str)
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
new_ECG_features = preprocessing_methods1.ECGPreprocessing1(ECG)
new_PPG_features, new_PPG_norm = preprocessing_methods1.PPGPreprocessing1(PPG)

# Run preprocessing methods for combining the data
new_df_init = preprocessing_methods1.combine_data1(new_ECG_features, new_PPG_features, 300)
new_df = preprocessing_methods1.features(new_df_init, new_PPG_norm)
new_df = new_df.replace([np.inf, -np.inf], np.nan).dropna()


# In[ ]:


scaler = joblib.load(config.model['sklearn_scaler_file_name'])
regressor = joblib.load(config.model['model_file_name'])


# In[ ]:


X = new_df.drop(list(new_df.filter(regex = '_idx')), axis=1)
#X.to_csv('X_test.csv')

y_pred = regressor.predict(scaler.transform(X))

pred_sys = [item[0] for item in y_pred]
pred_dias = [item[1] for item in y_pred]

print(pred_sys, pred_dias)

# In[ ]:


#fig,[ax1,ax2] = plt.subplots(2,1,figsize=(12,5),sharex=True)
#ax1.plot(pred_sys,'r-')
#ax1.set_title('Systolic BP')
#ax1.set_ylim([80,200])
#ax2.plot(pred_dias,'r-')
#ax2.set_title('Diastolic BP')
#ax2.set_ylim([40,160])
#plt.show()

