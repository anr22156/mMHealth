#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from enum import Enum


# In[ ]:


class SensorTypes(Enum):
    PPG = 'PPG'
    BP = 'BP'
    ECG = 'ECG'
        
    @staticmethod
    def list():
        return list(map(lambda c: c.value, SensorTypes))

