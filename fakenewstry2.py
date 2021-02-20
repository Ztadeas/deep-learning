import pandas as pd
import numpy as np
from keras import layers
from keras import models
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras import optimizers
import pickle
import os
import keras
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

real = pd.read_csv("C:\\Users\\Tadeas\\Downloads\\truenews\\True.csv")
fake = pd.read_csv("C:\\Users\\Tadeas\\Downloads\\fakenews\\Fake.csv")


text = []
subject_target = []
truefalse_target = []

rowstrue = 21417
rowsfalse = 23481

for i in range(rowstrue):
  subject_target.append(real["subject"][i])
  text.append(real["text"][i])
  truefalse_target.append("True")


for i in range(rowsfalse):
  subject_target.append(fake["subject"][i])
  text.append(fake["text"][i])
  truefalse_target.append("False")


num_subject = []
num_trueflase = []


for q in subject_target:
  if q == "politicsNews" or q == "politics" or q =="Government News":
    num_subject.append(1)

  else:
    num_subject.append(0)



for i in truefalse_target:
  if i == "True":
    num_trueflase.append(1)

  else:
    num_trueflase.append(0)



y_subject = np.asarray(num_subject, dtype="float32")
y_truefalse = np.asarray(num_trueflase, dtype="float32")


tokenizer = Tokenizer(num_words= len(text))
tokenizer.fit_on_texts(text)
seq = tokenizer.texts_to_sequences(text)
data = pad_sequences(seq, maxlen= 164)

split = 20000


realsamples = np.arange(data.shape[0])
np.random.shuffle(realsamples)
data = data[realsamples]
y_subject = y_subject[realsamples]
y_truefalse = y_truefalse[realsamples]


x_train = data[:split]
x_val = data[split:]
y_subject2 = y_subject[:split]
y_subject_val = y_subject[split:]
y_truefalse2 = y_truefalse[:split]
y_truefalse_val = y_truefalse[split:]



first = x_train.shape[0]

glove_dir = "C:\\Users\\Tadeas\\Downloads\\glove.6b"

embeddings_ind = {}
f = open(os.path.join(glove_dir, "glove.6B.100d.txt"), encoding="utf8")
for l in f:
  values = l.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype="float32")
  embeddings_ind[word] = coefs
f.close()

embeding_matrix = np.zeros((data.shape[0], 100))

for word, ind in tokenizer.word_index.items():
  if ind < data.shape[0]:
    embeding_vector = embeddings_ind.get(word)
    if embeding_vector is not None:
      embeding_matrix[ind] = embeding_vector



datas = keras.Input(shape=(x_train.shape[1:]))
embedded = layers.Embedding(data.shape[0], 100, input_length= 164, weights= [embeding_matrix], trainable = False)(datas)
x = layers.Conv1D(128, 5, activation="relu", padding="VALID")(embedded)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation="relu")(x)
x = layers.MaxPooling1D(3)(x)
x = layers.Conv1D(256, 5, activation="relu")(x)
x = layers.Conv1D(256, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation="relu")(x)



y_subject_prediction = layers.Dense(1, activation="sigmoid", name="subject")(x)
y_truefalse_prediction = layers.Dense(1, activation="sigmoid", name="truefalse")(x)

m = models.Model(datas, [y_subject_prediction, y_truefalse_prediction])


m.compile(optimizer=optimizers.Adam(lr = 0.001), loss={"subject": "binary_crossentropy", "truefalse": "binary_crossentropy"}, metrics=["acc"])

m.fit(x_train, {"subject": y_subject2, "truefalse": y_truefalse2}, epochs=100, batch_size=128, validation_data= (x_val, [y_subject_val, y_truefalse_val]))

m.save("fakenews5.h5")

with open('tokenizer_fakenews5.pickle', 'wb') as handle:
  pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)