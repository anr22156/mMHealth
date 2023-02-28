#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import config


# In[ ]:


scaler = joblib.load(config.model['sklearn_scaler_file_name'])
regressor = joblib.load(config.model['model_file_name'])


# In[ ]:


df = pd.read_csv(config.model['test_data_file'])
output = ['sys_val','dias_val']
y = df[output]
X = df.iloc[:,1:].drop(columns=output)


# In[ ]:


y_pred = regressor.predict(scaler.transform(X))
errors = abs(y_pred - y)
print(errors)

pred_sys = [item[0] for item in y_pred]
pred_dias = [item[1] for item in y_pred]


# In[ ]:


fig,[ax1,ax2] = plt.subplots(2,1,figsize=(12,5),sharex=True)
ax1.plot(pred_dias,'r-')
ax1.plot(y['dias_val'],'b--')
ax1.set_title('Diastolic BP')
ax1.legend(['pred','test'])
ax1.set_ylim([40,160])
ax2.plot(pred_sys,'r-')
ax2.plot(y['sys_val'],'b--')
ax2.legend(['pred','test'])
ax2.set_title('Systolic BP')
ax2.set_ylim([80,200])
plt.show()
#fig.savefig('test.png')

