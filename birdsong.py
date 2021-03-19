import librosa
import numpy as np
import os
from math import floor
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras import optimizers


dir_path = "C:\\Users\\Tadeas\\Downloads\\birdsongbig"

rozdeleni = 22050*3


def preprocess():
  mfccs = []
  labels = []
  
  for v, i in enumerate(os.listdir(dir_path)):
    near_full_path = os.path.join(dir_path, i)
    for x in os.listdir(near_full_path):
      full_path = os.path.join(near_full_path, x)
      signal, sr = librosa.load(full_path)
      for p in range(signal.shape[0] // rozdeleni):
        new_signal = signal[rozdeleni*p: rozdeleni*(p+1)]
        mfcc = librosa.feature.mfcc(new_signal, n_mfcc=13)
        mfccs.append(mfcc)
        labels.append(v)
    print(f"{v}: Done")

  mfccs = np.asarray(mfccs)

  return mfccs, labels        


mfccs, labels = preprocess()

print(mfccs.shape)

num_of_class = []

for e in os.listdir("C:\\Users\\Tadeas\\Downloads\\birdsongbig"):
  num_of_class.append(e)

labels = to_categorical(labels, num_classes=len(num_of_class))

realsamples = np.arange(data.shape[0])
np.random.shuffle(realsamples)
mfccs = mfccs[realsamples]
labels = labels[realsamples]


def train_val(data, labels):
  split = len(data)
  train_split = split * 0.8
  train_split = floor(train_split)
  train_data = data[:train_split]
  train_labels = labels[:train_split]
  val_data = data[train_split:]
  val_labels = labels[train_split:]

  return train_data, train_labels, val_data, val_labels

x, y, x_val, y_val = train_val(mfccs, labels)


m = models.Sequential()
m.add(layers.LSTM(64, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, input_shape=(x.shape[1], x.shape[2])))
m.add(layers.LSTM(128, dropout=0.1, recurrent_dropout=0.5))
m.add(layers.Dense(len(num_of_class), activation="softmax"))

m.compile(optimizer=optimizers.Adam(lr=0.001), loss="categorical_crossentropy", metrics=["acc"])

m.fit(x, y, batch_size=32, epochs=30, validation_data=(x_val, y_val))






