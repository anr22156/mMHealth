#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

from CleanupProcessor import CleanupProcessor
from DatapointsRepository import DatapointsRepository
from SensorTypeDatasourceMap import SensorTypeDatasourceMap

# In[ ]:


class PreparedDataProvider:
    def __init__(self, datapoints_repository: DatapointsRepository, sensor_type_datasource_map: SensorTypeDatasourceMap) -> None:
        self.__datapoints_repository = datapoints_repository
        self.__sensor_type_datasource_map = sensor_type_datasource_map
        
    def get(self, data_source: str):
        sensor_types = self.__sensor_type_datasource_map.get(data_source)
        cleanup_processor = CleanupProcessor(sensor_types)
        extracted_data = []
        datapoints = self.__datapoints_repository.get(data_source)
        extracted_data += datapoints
      
        dataframe = pd.DataFrame(extracted_data).set_index('_id')
        dataframe = dataframe.dropna()
        
        return dataframe #cleanup_processor.process(dataframe)

