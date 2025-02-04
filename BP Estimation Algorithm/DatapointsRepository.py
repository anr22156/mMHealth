#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import List

from pymongo import MongoClient, ASCENDING

from Sensor import Sensor
from SensorTypeDatasourceMap import SensorTypeDatasourceMap


# In[ ]:

# Defining the class to access the data collections stored locally in MongoDB

class DatapointsRepository:
    def __init__(self, mongo_client: MongoClient, sensor_type_datasource_map: SensorTypeDatasourceMap) -> None:
        self.__mongo_client = mongo_client
        self.__sensor_type_datasource_map = sensor_type_datasource_map
        
    def update(self, datasource: str, date, sensors: List[Sensor]):
        set_data = {sensor.type: sensor.value for sensor in sensors}
        set_data['date'] = date
        self.__get_client(datasource).update(
            {'_id': date.strftime('%m_%d_%Y_%H_%M_%S')},
            {'$set': set_data},
            upsert = True
        )
        
    def get(self, datasource: str) -> list:
        cursor = self.__get_client(datasource).find().sort("_id", ASCENDING)
        sensor_types = self.__sensor_type_datasource_map.get(datasource)
        
        return list(cursor)
            #filter(
            #lambda dp: True if all(sensor_type in dp for sensor_type in sensor_types) else False, cursor)
        #)
    
    # "PrenatalTracker" is the name of the database in MongoDB    
    def __get_client(self, datasource: str):
        return self.__mongo_client['PrenatalTracker'][datasource]

