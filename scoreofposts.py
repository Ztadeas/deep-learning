import pandas as pd 
import keras
from keras import models
from keras import layers
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import optimizers
from math import floor
import os

dir_path = "C:\\Users\Tadeas\\Downloads\\score of posts\\reddit_wsb.csv"

everything = pd.read_csv(dir_path)



text = []
scores = []



for i in range(34880):
  if pd.isna(everything["body"][i]) == True:  
    pass

  else:
    text.append(everything["body"][i])
    scores.append(everything["score"][i])

scores = np.asarray(scores, dtype="float64")


mean = scores.mean(axis=0)
scores -= mean
std = scores.std(axis=0)
scores = scores // std + 1

tokenizer = Tokenizer(num_words= len(text))
tokenizer.fit_on_texts(text)
seq = tokenizer.texts_to_sequences(text)
data = pad_sequences(seq, maxlen= 80)


realsamples = np.arange(data.shape[0])
np.random.shuffle(realsamples)
data = data[realsamples]
scores = scores[realsamples]

validatin_split = data.shape[0] * 0.8
validatin_split = floor(validatin_split)

x_train = data[:validatin_split]
y_train = scores[:validatin_split]
x_val = data[validatin_split:]
y_val = scores[validatin_split:]

print(y_train)

glove_dir = "C:\\Users\\Tadeas\\Downloads\\glove.6b"

embeddings_ind = {}
f = open(os.path.join(glove_dir, "glove.6B.100d.txt"), encoding="utf8")
for l in f:
  values = l.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype="float32")
  embeddings_ind[word] = coefs
f.close()

embeding_matrix = np.zeros((len(data), 100))

for word, ind in tokenizer.word_index.items():
  if ind < len(data):
    embeding_vector = embeddings_ind.get(word)
    if embeding_vector is not None:
      embeding_matrix[ind] = embeding_vector


m = models.Sequential()
m.add(layers.Embedding(data.shape[0], 100, input_length=80))
m.add(layers.Conv1D(64, 7, padding="same", activation="relu"))
m.add(layers.MaxPooling1D(5))
m.add(layers.Conv1D(128, 7, padding="same", activation="relu"))
m.add(layers.MaxPooling1D(3))
m.add(layers.Conv1D(256, 7, padding="same", activation="relu"))
m.add(layers.Conv1D(256, 7, padding="same", activation="relu"))
m.add(layers.MaxPooling1D(3))
m.add(layers.Conv1D(256, 7, padding="same", activation="relu"))
m.add(layers.GlobalMaxPooling1D())

m.add(layers.Dense(1))

m.layers[0].set_weights([embeding_matrix])
m.layers[0].trainable = False

m.compile(optimizer= optimizers.Adam(lr= 0.001), loss= "mse", metrics=["acc"])

m.fit(x_train, y_train, epochs=30, batch_size= 32, validation_data=(x_val, y_val))

m.save("scoreofpostspredict.h5")














  