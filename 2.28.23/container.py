#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import Callable

from pymongo import MongoClient

import config
from ModelBuilder import ModelBuilder
from PreparedDataProvider import PreparedDataProvider
from DatapointsRepository import DatapointsRepository
from Serial import Serial
from SensorBuilder import SensorBuilder
from SensorTypeDatasourceMap import SensorTypeDatasourceMap


# In[ ]:


def singleton(function: Callable):
    caching = {}
    def wrapper(*args, **kwargs):
        if function.__name__ in caching:
            return caching[function.__name__]
        caching[function.__name__] = function(*args, **kwargs)

        return caching[function.__name__]

    return wrapper


# In[ ]:


class Container:
    
    @singleton
    def model_builder(self) -> ModelBuilder:
        return ModelBuilder()
   
    @singleton
    def mongo_client(self) -> MongoClient:
        return MongoClient(config.mongodb['host'],config.mongodb['port'])
    
    @singleton
    def prepared_data_provider(self) -> PreparedDataProvider:
        return PreparedDataProvider(self.datapoints_repository(), self.sensor_type_datasource_map())
    
    @singleton
    def datapoints_repository(self) -> DatapointsRepository:
        return DatapointsRepository(self.mongo_client(), self.sensor_type_datasource_map())

    @singleton
    def serial(self) -> Serial:
        return Serial(config.serial['port'], config.serial['baud_rate'])

    @singleton
    def sensor_builder(self) -> SensorBuilder:
        return SensorBuilder()

    @singleton
    def sensor_type_datasource_map(self) -> SensorTypeDatasourceMap:
        return SensorTypeDatasourceMap()    

