from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

i = "Math Professor Scott Steiner says the numbers spell DISASTER for Gamestop shorts"

m = load_model("scoreofpostspredict.h5")

tokenizer = Tokenizer(num_words= len(i))
tokenizer.fit_on_texts(i)
seq = tokenizer.texts_to_sequences(i)
data = pad_sequences(seq, maxlen= 80)

k = m.predict(data)

print(np.argmax(k))