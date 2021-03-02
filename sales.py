import pandas as pd
import numpy as np
import math
from keras import Input
from keras import models
from keras import layers
from keras import optimizers

dir_path = "C:\\Users\\Tadeas\\Downloads\\housesales\\kc_house_data.csv"

everything = pd.read_csv(dir_path)

print(everything)

bedrooms = []
bathrooms = []
sqft_living = []
floor = []
labels = []


for i in range(21613):
  bedrooms.append(everything["bedrooms"][i])
  bathrooms.append(everything["bathrooms"][i])
  sqft_living.append(everything["sqft_living"][i])
  floor.append(everything["floors"][i])
  labels.append(everything["price"][i])


def toarray(data):
  x = np.asarray(data, dtype= "float32")
  x = np.reshape(x, (len(data), 1))
  return x

bedrooms = toarray(bedrooms)

bathrooms = toarray(bathrooms)

sqft_living = toarray(sqft_living)

floor = toarray(floor)

labels = toarray(labels)

def normalize(data):
  mean = data.mean(axis=0)
  data -= mean
  std = data.std(axis=0)
  data = data / std + 1

  return data

bedrooms = normalize(bedrooms)

bathrooms = normalize(bathrooms)

sqft_living = normalize(sqft_living)

floor = normalize(floor)

labels = normalize(labels)


def validation_split(data):
  split = len(data) * 0.95
  split = math.floor(split)
  train = data[:split]
  val = data[split:]
  return train, val

bedrooms_train, bedrooms_val = validation_split(bedrooms)

bathrooms_train, bathrooms_val = validation_split(bathrooms)

sqft_living_train, sqft_living_val = validation_split(sqft_living)

floor_train, floor_val = validation_split(floor)

labels_train, labels_val = validation_split(labels)

bedrooms_input = Input(shape=(1,), dtype="float32")
x_bedrooms = layers.Dense(128, activation="relu")(bedrooms_input)
x_bedrooms = layers.Dense(256, activation="relu")(x_bedrooms)
x_bedrooms = layers.Dense(256, activation="relu")(x_bedrooms)

bathrooms_input = Input(shape=(1,), dtype="float32")
x_bathrooms = layers.Dense(128, activation="relu")(bathrooms_input)
x_bathrooms = layers.Dense(256, activation="relu")(x_bathrooms)
x_bathrooms = layers.Dense(256, activation="relu")(x_bathrooms)

sqft_living_input = Input(shape=(1,), dtype="float32")
x_living = layers.Dense(128, activation="relu")(sqft_living_input)
x_living = layers.Dense(256, activation="relu")(x_living)
x_living = layers.Dense(256, activation="relu")(x_living)

floor_input = Input(shape=(1,), dtype="float32")
x_floor = layers.Dense(128, activation="relu")(floor_input)
x_floor = layers.Dense(256, activation="relu")(x_floor)
x_floor = layers.Dense(256, activation="relu")(x_floor)

cocn = layers.concatenate([x_bedrooms, x_bathrooms, x_living, x_floor], axis= -1)

answer = layers.Dense(1)(cocn)

m = models.Model([bedrooms_input, bathrooms_input, sqft_living_input, floor_input], answer)

m.compile(optimizer=optimizers.SGD(lr=0.001), loss= "mse", metrics= ["mae"])

m.fit([bedrooms_train, bathrooms_train, sqft_living_train, floor_train], labels_train, epochs=30, batch_size=64, validation_data=([bedrooms_val, bathrooms_val, sqft_living_val, floor_val], labels_val))

m.save("predicthouses.h5")



  
  

  
  





  
  




  

  



  


  

  




