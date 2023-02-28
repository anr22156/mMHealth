#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import datetime

from container import Container
from DataSource import DataSource


# In[ ]:


container = Container()
serial_device = container.serial()
serial_device.connect()
sensor_builder = container.sensor_builder()
datapoint_repository = container.datapoints_repository()


# In[ ]:


while True:
    serial_device.listen(sensor_builder.add_text)
    if sensor_builder.is_complete():
        sensors = sensor_builder.build()
        print(sensors)
        datapoint_repository.update(DataSource.COLLECTION2.value, datetime.datetime.now(), sensors)
        
serial_device.disconnect()

