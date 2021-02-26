from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras import models
from keras.optimizers import RMSprop


max_fetures = 10000
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_fetures)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)

x_test = sequence.pad_sequences(x_test, maxlen=max_len)

m = models.Sequential()
m.add(layers.Embedding(max_fetures, 128, input_length=max_len))
m.add(layers.Conv1D(32, 7, activation="relu"))
m.add(layers.MaxPooling1D(5))
m.add(layers.Conv1D(32, 7, activation="relu"))
m.add(layers.GlobalMaxPooling1D())
m.add(layers.Dense(1))

m.compile(optimizer=RMSprop(learning_rate=1e-4), loss= "binary_crossentropy", metrics=["acc"])

m.fit(x_train, y_train, epochs=10, batch_size=128, validation_split= 0.2)


