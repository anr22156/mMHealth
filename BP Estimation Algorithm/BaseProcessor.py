#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import abc


# In[ ]:


class BaseProcessor(metaclass=abc.ABCMeta):
    def process(self, dataframe):
        pass

