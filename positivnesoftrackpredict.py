import pandas as pd 
import numpy as np
from keras.models import Model
from keras import layers
from keras import Input
from keras import optimizers
from math import floor

dir_path = "C:\\Users\\Tadeas\\Downloads\\track\\data.csv"

everything = pd.read_csv(dir_path)

everything = everything[0:170000]

ranges = 170000

acoust = []
dance = []
instrument = []
energy = []
explicit = []
live = []
speech = []
tempo = []
valence = []


popularity = []
year = []


for i in range(ranges):
  acoust.append(everything["acousticness"][i])
  dance.append(everything["danceability"][i])
  energy.append(everything["energy"][i])
  explicit.append(everything["explicit"][i])
  instrument.append(everything["instrumentalness"][i])
  live.append(everything["liveness"][i])
  speech.append(everything["speechiness"][i])
  tempo.append(everything["tempo"][i])
  valence.append(everything["valence"][i])
  popularity.append(everything["popularity"][i])
  year.append(everything["year"][i])


acoust = np.asarray(acoust, dtype="float32")
dance = np.asarray(dance, dtype="float32")
instrument = np.asarray(instrument, dtype="float32")
energy = np.asarray(energy, dtype="float32")
explicit = np.asarray(explicit, dtype="float32")
live = np.asarray(live, dtype="float32")
speech = np.asarray(speech, dtype="float32")
tempo = np.asarray(tempo, dtype="float32")
valence = np.asarray(valence, dtype="float32")


popularity = np.asarray(popularity, dtype="float32")
year = np.asarray(year, dtype="float32")

acoust = np.reshape(acoust, (ranges, 1))
dancen = np.reshape(dance, (ranges, 1))
instrument = np.reshape(instrument, (ranges, 1))
energy = np.reshape(energy, (ranges, 1))
explicit = np.reshape(explicit, (ranges, 1))
live = np.reshape(live, (ranges, 1))
speech = np.reshape(speech, (ranges, 1))
tempo = np.reshape(tempo, (ranges, 1))
valence = np.reshape(valence, (ranges, 1))


popularity = np.reshape(popularity, (ranges, 1))
year = np.reshape(year, (ranges, 1))

mean = tempo.mean(axis=0)
tempo -= mean
std = tempo.std(axis=0)
tempo = tempo // std + 1

mean = year.mean(axis=0)
year -= mean
std = year.std(axis=0)
year = year // std + 1

mean = popularity.mean(axis=0)
popularity -= mean
std = popularity.std(axis=0)
popularity = popularity // std + 1

print(tempo)
print(popularity)
print(year)

validation_split = len(year) * 0.9
validation_split = floor(validation_split)

acoust_train = acoust[:validation_split]
acoust_val = acoust[validation_split:]
dance_train = dance[:validation_split]
dance_val = dance[validation_split:]
instrument_train = instrument[:validation_split]
instrument_val = instrument[validation_split:]
energy_train = energy[:validation_split]
energy_val = energy[validation_split:]
explicit_train = explicit[:validation_split]
explicit_val = explicit[validation_split:]
live_train = live[:validation_split]
live_val = live[validation_split:]
speech_train = speech[:validation_split]
speech_val = speech[validation_split:]
tempo_train = tempo[:validation_split]
tempo_val = tempo[validation_split:]
valence_train = valence[:validation_split]
valence_val = valence[validation_split:]
popularity_train = popularity[:validation_split]
popularity_val = popularity[validation_split:]
year_train = year[:validation_split]
year_val = year[validation_split:]


acoust_input = Input(shape=(1,), dtype= "float32")
dance_input = Input(shape=(1,), dtype= "float32")
instrument_input = Input(shape=(1,), dtype= "float32")
energy_input = Input(shape=(1,), dtype= "float32")
explicit_input = Input(shape=(1,), dtype= "float32")
live_input = Input(shape=(1,), dtype= "float32")
speech_input = Input(shape=(1,), dtype= "float32")
tempo_input = Input(shape=(1,), dtype= "float32")
valence_input = Input(shape=(1,), dtype= "float32")

acoust_x = layers.Dense(128, activation="relu")(acoust_input)
acoust_x = layers.Dense(256, activation="relu")(acoust_x)
acoust_x = layers.Dense(256, activation="relu")(acoust_x)
acoust_x = layers.Dense(256, activation="relu")(acoust_x)

dance_x = layers.Dense(128, activation="relu")(dance_input)
dance_x = layers.Dense(256, activation="relu")(dance_x)
dance_x = layers.Dense(256, activation="relu")(dance_x)
dance_x = layers.Dense(256, activation="relu")(dance_x)

instrument_x = layers.Dense(128, activation="relu")(instrument_input)
instrument_x = layers.Dense(256, activation="relu")(instrument_x)
instrument_x = layers.Dense(256, activation="relu")(instrument_x)
instrument_x = layers.Dense(256, activation="relu")(instrument_x)

explicit_x = layers.Dense(128, activation="relu")(explicit_input)
explicit_x = layers.Dense(256, activation="relu")(explicit_x)
explicit_x = layers.Dense(256, activation="relu")(explicit_x)
explicit_x = layers.Dense(256, activation="relu")(explicit_x)

live_x = layers.Dense(128, activation="relu")(live_input)
live_x = layers.Dense(256, activation="relu")(live_input)
live_x = layers.Dense(256, activation="relu")(live_input)
live_x = layers.Dense(256, activation="relu")(live_input)

energy_x = layers.Dense(128, activation="relu")(energy_input)
energy_x = layers.Dense(256, activation="relu")(energy_x)
energy_x = layers.Dense(256, activation="relu")(energy_x)
energy_x = layers.Dense(256, activation="relu")(energy_x)

speech_x = layers.Dense(128, activation="relu")(speech_input)
speech_x = layers.Dense(256, activation="relu")(speech_x)
speech_x = layers.Dense(256, activation="relu")(speech_x)
speech_x = layers.Dense(256, activation="relu")(speech_x)

tempo_x = layers.Dense(128, activation="relu")(tempo_input)
tempo_x = layers.Dense(256, activation="relu")(tempo_x)
tempo_x = layers.Dense(256, activation="relu")(tempo_x)
tempo_x = layers.Dense(256, activation="relu")(tempo_x)

valence_x = layers.Dense(128, activation="relu")(valence_input)
valence_x = layers.Dense(256, activation="relu")(valence_x)
valence_x = layers.Dense(256, activation="relu")(valence_x)
valence_x = layers.Dense(256, activation="relu")(valence_x)

conc = layers.concatenate([acoust_x, dance_x, instrument_x, explicit_x, live_x, energy_x, speech_x, tempo_x, valence_x], axis= -1)

popular_pred = layers.Dense(1)(conc)
year_pred = layers.Dense(1)(conc)

m = Model([acoust_input, dance_input, instrument_input, explicit_input, live_input, energy_input, speech_input, tempo_input, valence_input], [popular_pred, year_pred])

m.compile(optimizer=optimizers.SGD(lr=0.001), loss = ["mse", "mse"], metrics= ["acc"])

m.fit([acoust_train, dance_train, instrument_train, explicit_train, live_train, energy_train, speech_train, tempo_train, valence_train], [popularity_train , year_train], epochs = 30, batch_size = 64, validation_split=0.2)

























  
  