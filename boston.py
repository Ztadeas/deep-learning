from keras.datasets import boston_housing
import numpy as np
from keras import layers
from keras import models

(train_data, train_targets), (test_data, test_target) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

def build_model():
  m = models.Sequential()
  m.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
  m.add(layers.Dense(64, activation="relu"))
  m.add(layers.Dense(1))
  m.compile(optimizer = "rmsprop", loss = "mse", metrics = ["mae"])
  return m

k = 4
num_val_samples = len(train_data) // k
nums_epochs = 100
all_scores = []

for i in range(k):
    print("processing", i)
    val_data = train_data[i*num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i*num_val_samples: (i+1) * num_val_samples]

    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i+1) * num_val_samples:]], axis=0)

    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i+1) * num_val_samples:]], axis = 0)

    m = build_model()
    m.fit(partial_train_data, partial_train_targets, epochs= nums_epochs, batch_size= 1, verbose= 0)
    val_mse, val_mae = m.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))

m = build_model()

m.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = m.evaluate(test_data, test_target)

print(test_mae_score)
