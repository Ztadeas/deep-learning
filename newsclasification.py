import os
import numpy as np
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import optimizers
import tensorflow as tf
from keras.losses import Reduction

text = []
label = []


dir_path = "C:\\Users\\Tadeas\\Downloads\\bbcclassifier\\bbc"

for i in ["business", "entertainment", "politics", "sport", "tech"]:
  full_path = os.path.join(dir_path, i)
  for l in os.listdir(full_path):
    k = os.path.join(full_path, l)
    with open(k, "r") as f:
      text.append(f.read())

    if i == "business":
      label.append(0)

    elif i == "entertainment":
      label.append(1)

    elif i == "politics":
      label.append(2)

    elif i == "sport":
      label.append(3)

    elif i == "tech":
      label.append(4)

    else:
      pass


maxlens = 164
label = np.asarray(label)

label = to_categorical(label, num_classes= 5)

tokenizer = Tokenizer(num_words= len(text))
tokenizer.fit_on_texts(text)
seq = tokenizer.texts_to_sequences(text)
data = pad_sequences(seq, maxlen= maxlens)


realsamples = np.arange(data.shape[0])
np.random.shuffle(realsamples)
data = data[realsamples]
label = label[realsamples]

print(label)

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
m.add(layers.Embedding(len(data), 100, input_length=maxlens))
m.add(layers.Conv1D(128, 5, activation="relu"))
m.add(layers.MaxPooling1D(5))
m.add(layers.Conv1D(256, 5, activation="relu"))
m.add(layers.Conv1D(256, 5, activation="relu"))
m.add(layers.MaxPooling1D(3))
m.add(layers.Conv1D(256, 5, activation="relu"))
m.add(layers.GlobalMaxPooling1D())
m.add(layers.Dropout(0.5))
m.add(layers.Dense(5, activation="softmax"))


m.layers[0].set_weights([embeding_matrix])
m.layers[0].trainable = False

m.compile(optimizer= optimizers.Adam(lr= 0.001), loss= tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0,name= "categorical_crossentropy"), metrics= ["acc"])


m.fit(data, label, epochs= 30, batch_size= 16, validation_split= 0.2)

m.save("newsprediction10.h5")
