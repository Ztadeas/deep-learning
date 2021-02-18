import keras 
import numpy as np
from keras import layers
from keras import models
import random
import sys


path = keras.utils.get_file("nietzsche.txt", origin= "https://s3.amazonaws.com/text-datasets/nietzsche.txt")

text = open(path).read().lower()

maxlen = 60

step = 3

next_chars = []

sentences = []

for i in range(0, len(text) - maxlen, step):
  sentences.append(text[i: i +maxlen])
  next_chars.append(text[i + maxlen])

chars = sorted(list(set(text)))

char_indices = dict((char, chars.index(char)) for char in chars)

x = np.zeros((len(sentences), maxlen, len(chars)), dtype= np.bool)

y = np.zeros((len(sentences), len(chars)))

for i, sentence in enumerate(sentences):
  for t, char in enumerate(sentence):
    x[i, t, char_indices[char]] = 1
    y[i , char_indices[next_chars[i]]] = 1

m = models.Sequential()
m.add(layers.LSTM(128, input_shape = (maxlen, len(chars))))
m.add(layers.Dense(len(chars), activation="softmax"))

m.compile(loss="categorical_crossentropy", optimizer= keras.optimizers.Adam(lr = 0.001))

def sample(preds, temp = 1.0):
  preds = np.asarray(preds).astype("float64")
  preds = np.log(preds) / temp
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

for epoch in range(1, 60):
  print("Epoch", epoch)
  m.fit(x, y, batch_size=128, epochs=1)
  start_index = random.randint(0, len(text) - maxlen - 1)
  gentext = text[start_index: start_index + maxlen]

  for temp in [0.2, 0.5, 1.0, 1.2]:
    print("temp", temp)
    sys.stdout.write(gentext)

    for i in range(400):
      sampled = np.zeros((1, maxlen, len(chars)))
      for t, char in enumerate(gentext):
        sampled[0, t, char_indices[char]] = 1

      preds = m.predict(sampled, verbose=0)[0]
      next_index = sample(preds, temp)
      next_char = chars[next_index]
      gentext += next_char
      gentext = gentext[1:]
      sys.stdout.write(next_char)



