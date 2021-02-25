import pandas as pd 
import random
import numpy as np
from keras.models import Model
from keras import layers
from keras import Input
from keras import optimizers
from math import floor

dir_path = "C:\\Users\\Tadeas\\Downloads\\titanic\\train.csv"
valdir = "C:\\Users\\Tadeas\\Downloads\\titanictest\\test.csv"


everything = pd.read_csv(dir_path)
validation = pd.read_csv(valdir)


print(everything)

labels = []
sex = []
age = []
pclass = []


for i in range(891):
  labels.append(everything["Survived"][i])
  pclass.append(everything["Pclass"][i])
  if pd.isna(everything["Age"][i]) == True:
    r = random.randint(18, 55)
    age.append(r)

  else:
    age.append(everything["Age"][i])

  if everything["Sex"][i] == "male":
    sex.append(1)

  else:
    sex.append(0)



labels = np.asarray(labels, dtype="float64")
sex = np.asarray(sex, dtype="float64")
age = np.asarray(age, dtype="float64")
pclass = np.asarray(pclass, dtype="float64")

sex = np.reshape(sex, (891, 1))
age = np.reshape(age, (891, 1))
pclass = np.reshape(pclass, (891, 1))  

mean = age.mean(axis=0)
age -= mean
std = age.std(axis=0)
age = age // std + 2

validation_split = len(sex) * 0.8
validation_split = floor(validation_split)

val_labels = labels[validation_split:]
train_labels = labels[:validation_split]
val_sex = sex[validation_split:]
train_sex = sex[:validation_split]
val_age = age[validation_split:]
train_age = age[:validation_split]
val_pclass = pclass[validation_split:]
train_pclass = pclass[:validation_split]

print(len(train_labels))
print(len(val_labels))

age_input = Input(shape=(1,), dtype="float64", name="age")
sex_input = Input(shape=(1,), dtype="float64", name="sex")
pclass_input = Input(shape=(1,), dtype="float64", name="pclass")

x_age = layers.Dense(64, activation= "relu")(age_input)
x_age = layers.Dense(128, activation= "relu")(x_age)
x_age = layers.Dense(256, activation="relu")(x_age)

x_sex = layers.Dense(64, activation= "relu")(sex_input)
x_sex = layers.Dense(128, activation= "relu")(x_sex)
x_sex = layers.Dense(256, activation= "relu")(x_sex)

x_pclass = layers.Dense(64, activation= "relu")(pclass_input)
x_pclass = layers.Dense(128, activation= "relu")(x_pclass)
x_pclass = layers.Dense(256, activation= "relu")(x_pclass)

conc = layers.concatenate([x_age, x_sex, x_pclass], axis = -1)

finall = layers.Dense(1, activation="sigmoid")(conc)

m = Model([age_input, sex_input, pclass_input], finall)

m.compile(optimizer= optimizers.Adam(lr=0.001), loss="binary_crossentropy", metrics=["acc"])

m.fit({"age": train_age, "sex": train_sex, "pclass": train_pclass}, train_labels, epochs = 30, batch_size= 8, validation_data=([val_age, val_sex, val_pclass], val_labels))







