import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Input

CWD = os.getcwd()

data_dir = CWD+'/csv/test.csv'


df = pd.read_csv(data_dir)

data = df[10:20]
print(data)
 
data = np.array(data, dtype=np.float32)
data = np.reshape(data,(-1,10,13))
print(data.shape)
model= load_model('new_trained_all_columns_dropout.h5')

y = model.predict(data)

y = np.array(y)
y = y[:,[12]]

if (y >0) & (y<1.5):
    print(y,"The process is: Machining")
elif (y>1) & (y<2.5):
    print(y,"The process is:Drawer Open")
elif (y>2.5) &(y<3.5):
    print(y,"The process is: Drawe Close")
else:
    print('none')