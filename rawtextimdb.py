import os
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras import layers

imdb_dir = "C:\\Users\\Tadeas\\Downloads\\aclImdb_v1\\aclImdb"
train_dir = os.path.join(imdb_dir, "train")

labels = []
texts = []

for l in ["neg", "pos"]:
  dir_name = os.path.join(train_dir, l)
  for name in os.listdir(dir_name):
    if name[-4:] == ".txt":
      f = open(os.path.join(dir_name, name), encoding= "utf8")
      texts.append(f.read())
      f.close()
     
      if l == "neg":
        labels.append(0)

      else:
        labels.append(1)


maxlen = 100
training_samples = 200
validation_data = 10000
max_words = 10000

tokenizer = Tokenizer(num_words= max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_data]
y_val = labels[training_samples: training_samples + validation_data]

glove_dir = "C:\\Users\\Tadeas\\Downloads\\glove.6b"

embeddings_ind = {}
f = open(os.path.join(glove_dir, "glove.6B.100d.txt"), encoding="utf8")
for l in f:
  values = l.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype="float32")
  embeddings_ind[word] = coefs
f.close()

embeding_matrix = np.zeros((max_words, 100))

for word, ind in tokenizer.word_index.items():
  if ind < max_words:
    embeding_vector = embeddings_ind.get(word)
    if embeding_vector is not None:
      embeding_matrix[ind] = embeding_vector

m = Sequential()
m.add(Embedding(max_words, 100, input_length=maxlen))
m.add(Flatten())
m.add(Dense(32, activation="relu"))
m.add(Dense(1, activation="sigmoid"))
m.summary()

m.layers[0].set_weights([embeding_matrix])
m.layers[0].trainable = False


m.compile(optimizer= "rmsprop", loss= "binary_crossentropy", metrics = ["acc"])

m.fit(x_train, y_train, epochs= 10, batch_size = 32, validation_data=(x_val, y_val))

m.save_weights("pretrainedglovemodel.h5")




