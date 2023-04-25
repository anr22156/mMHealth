#!/usr/bin/env python
# coding: utf-8

# In[ ]:


mongodb = {
    'host': 'localhost',
    'port': 27017
}

serial = {
    'port': '',
    'baud_rate': 9600    
}

# The names of the files that the model, etc. will save to once trained
model = {
    'model_file_name': 'sample_data_PrenatalTracker.h5',
    'sklearn_scaler_file_name': 'sample_data_scaler.save',
    'test_data_file': 'sample_data_test_data.csv'
}

