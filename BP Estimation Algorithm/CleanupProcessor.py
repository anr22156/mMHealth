#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from BaseProcessor import BaseProcessor


# In[ ]:


class CleanupProcessor(BaseProcessor):
    def __init__(self, sensor_names: list) -> None:
        self.__sensor_names = sensor_names
        
    def process(self, dataframe):
        cols = [col_name for col_name in dataframe.columns]
        dataframe = dataframe[cols]
        #to_remove = [sensor + '_' + feature for sensor in self.__sensor_names for feature in self.__feature_names]
        
        return dataframe #.drop(to_remove, axis=1, errors='ignore')

