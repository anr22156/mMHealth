#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
import math

import config


# In[ ]:


scaler = joblib.load(config.model['sklearn_scaler_file_name'])
regressor = joblib.load(config.model['model_file_name'])


# In[ ]:


df = pd.read_csv(config.model['test_data_file'])
output = ['sys_val','dias_val']
y = df[output]
X = df.iloc[:,1:].drop(['sys_val','dias_val'],axis=1)


# In[ ]:


y_pred = regressor.predict(scaler.transform(X))
#errors = abs(y_pred - y)
#print(errors)

pred_sys = [item[0] for item in y_pred]
pred_dias = [item[1] for item in y_pred]

print("RMSE for SBP", math.sqrt(mse(y['sys_val'], pred_sys)))
print("RMSE for DBP", math.sqrt(mse(y['dias_val'], pred_dias)))

# In[ ]:


fig,[ax1,ax2] = plt.subplots(2,1,figsize=(12,5),sharex=True)
ax1.plot(pred_sys[:100],'r-')
ax1.plot(y['sys_val'][:100],'b--')
ax1.legend(['estimated','recorded'])
ax1.set_title('Systolic BP', fontsize=10)
ax1.set_ylim([80,200])
ax2.plot(pred_dias[:100],'r-')
ax2.plot(y['dias_val'][:100],'b--')
ax2.set_title('Diastolic BP', fontsize=10)
ax2.legend(['estimated','recorded'])
ax2.set_ylim([40,160])
txt = 'Comparison of the estimated and recorded BP values'
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.show()
#fig.savefig('test.png')

