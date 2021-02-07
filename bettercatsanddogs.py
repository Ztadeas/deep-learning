from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from keras import models
from keras import layers
from keras import optimizers



conv_base = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))


base_dir = "C:\\Users\\Tadeas\\Downloads\\cats_and_dogs"

train_dir = os.path.join(base_dir, "train")

val_dir = os.path.join(base_dir, "validation")

test_dir = os.path.join(base_dir, "test")

datagen = ImageDataGenerator(rescale= 1./255)
batch_size = 20

def extract_features(directory, sample_count):
  features = np.zeros(shape= (sample_count, 4, 4, 512))
  labels = np.zeros(shape= (sample_count))
  generator = datagen.flow_from_directory(directory, target_size= (150, 150), batch_size= batch_size, class_mode= "binary")
  i = 0
  for inputs_batch, labels_batch in generator:
    features_batch = conv_base.predict(inputs_batch)
    features[i * batch_size: (i+1) * batch_size] = features_batch
    labels[i * batch_size: (i+1) * batch_size] = labels_batch

    i += 1

    if i*batch_size >= sample_count:
      break

  return features, labels

train_features, trlab = extract_features(train_dir, 2000) 
val_features, vallab = extract_features(val_dir, 1000)
test_features , test_labels = extract_features(test_dir, 1000)

train_features = np.reshape(train_features, (2000, 4*4*512))
val_features = np.reshape(val_features, (1000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))

m = models.Sequential()

m.add(layers.Dense(256, activation="relu", input_dim = 4*4*512))
m.add(layers.Dropout(0.5))
m.add(layers.Dense(1, activation="sigmoid"))

m.compile(loss= "binary_crossentropy", optimizer= optimizers.RMSprop(lr=1e-4), metrics=["acc"])

m.fit(train_features, trlab, epochs= 20, batch_size= 20, validation_data= (val_features, vallab))

test_loss, test_acc = m.evaluate(test_features, test_labels)
print(test_acc)

