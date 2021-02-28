import pandas as pd 
import numpy as np
import math 
from keras import models
from keras import layers
from keras import optimizers

dir_path = "C:\\Users\\Tadeas\\Downloads\\stockpredictio\\TSLA.csv"

everything = pd.read_csv(dir_path)

features = []


for i in range(2416):
  features.append([everything["Open"][i], everything["High"][i], everything["Low"][i], everything["Close"][i]])


features = np.asarray(features, dtype="float32")

mean = features.mean(axis=0)
features -= mean

std = features.std(axis=0)
features = features / std + 2

def preproces(percentage, data):
  split = len(features) * percentage
  split = math.floor(split)
  train_data = data[:split]
  test_data = data[split:]

  return train_data, test_data

train, test = preproces(0.8, features)


def timeseries(data, timestep):
  x_data = []
  y_data = []
  for i in range(len(data) - timestep - 1):
    x_data.append(data[i: (i+ timestep), :])
    y_data.append(data[i+timestep, 3])

  return x_data, y_data

x_train, y_train = timeseries(train, 7)
x_test, y_test = timeseries(test, 7)

x_train = np.asarray(x_train, dtype="float32")
y_train = np.asarray(y_train, dtype= "float32")
    
x_test = np.asarray(x_test, dtype="float32")
y_test = np.asarray(y_test, dtype="float32")


m = models.Sequential()

m.add(layers.LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.5, input_shape= (7, 4)))
m.add(layers.LSTM(64, activation="relu", dropout= 0.1, recurrent_dropout= 0.5))
m.add(layers.Dense(1))

m.compile(optimizer=optimizers.Adam(lr=0.001), loss="mse", metrics= ["mae"])

m.fit(x_train, y_train, epochs= 30, batch_size= 16, validation_data=(x_test, y_test))

    
k = m.predict(x_train)    

print(np.argmax(k))



  
  



    



  

  

