#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from DataSource import DataSource
from SensorTypes import SensorTypes


# In[ ]:


class SensorTypeDatasourceMap:
    MAP = {
        DataSource.COLLECTION1.value: [
            SensorTypes.PPG.value, SensorTypes.BP.value, SensorTypes.ECG.value
        ],
        DataSource.COLLECTION2.value: [
            SensorTypes.PPG.value, SensorTypes.ECG.value,
        ],
    }
    
    def get(self, datasource: str) -> list:
        if datasource not in self.MAP:
            raise Exception('Mapping for datasource {0} not found'.format(datasource))
        return self.MAP[datasource]

