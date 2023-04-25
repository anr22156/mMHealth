#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


class ModelBuilder:
    def build(self):
        model = RandomForestRegressor(n_estimators = 100, random_state = 42)
        return model

