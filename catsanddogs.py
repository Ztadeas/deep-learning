import sys
import os
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

base_dir = "C:\\Users\\Tadeas\\Downloads\\cats_and_dogs"


train_dir = os.path.join(base_dir, "train")

valid_dir = os.path.join(base_dir, "validation")

test_dir = os.path.join(base_dir, "test")

train_cats_dir = os.path.join(train_dir, "cats")

train_dogs_dir = os.path.join(train_dir, "dogs")

val_cats_dir = os.path.join(valid_dir, "cats")

val_dogs_dir = os.path.join(valid_dir, "dogs")

test_cats_dir = os.path.join(test_dir, "cats")

test_dogs_dir = os.path.join(test_dir, "dogs")

m = models.Sequential()

m.add(layers.Conv2D(32, (3, 3), activation= "relu", input_shape= (150, 150, 3)))
m.add(layers.MaxPooling2D((2, 2)))
m.add(layers.Conv2D(64, (3, 3), activation= "relu"))
m.add(layers.MaxPooling2D((2, 2)))
m.add(layers.Conv2D(128, (3, 3), activation= "relu"))
m.add(layers.MaxPooling2D((2, 2)))
m.add(layers.Conv2D(128, (3, 3), activation= "relu"))
m.add(layers.MaxPooling2D((2, 2)))
m.add(layers.Conv2D(128, (3, 3), activation= "relu"))
m.add(layers.MaxPooling2D((2, 2)))
m.add(layers.Flatten())
m.add(layers.Dropout(0.5))
m.add(layers.Dense(512, activation= "relu"))
m.add(layers.Dense(1, activation = "sigmoid"))

m.compile(loss= "binary_crossentropy", optimizer= optimizers.RMSprop(lr=1e-4), metrics=["acc"])

train_datagen = ImageDataGenerator(rescale= 1./255, rotation_range= 60, width_shift_range= 0.5, height_shift_range= 0.5, shear_range= 0.5, zoom_range= 0.5, horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size= (150, 150), batch_size= 32, class_mode= "binary")

val_generator = test_datagen.flow_from_directory(valid_dir, target_size= (150, 150), batch_size= 32, class_mode= "binary" )

history = m.fit_generator(train_generator, steps_per_epoch= 100, epochs= 100, validation_data= val_generator, validation_steps= 50)

m.save("Cats_and_dogs_2.h5")



