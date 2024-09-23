import io
import random

import numpy as np
import tensorflow as tf
import keras
from keras import layers

# Model / data parameters
num_classes = 3
input_shape = (128, 128, 3)
batch_size = 32
epochs = 500

# Load the data and split it between train and test sets
train_dataset = keras.utils.image_dataset_from_directory(
    directory='images/128x128/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(input_shape[0], input_shape[1])
)

test_dataset = keras.utils.image_dataset_from_directory(
    directory='images/128x128/test',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(input_shape[0], input_shape[1])
)

# Data normalization
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(height_factor=0.2, width_factor=0.2)
])
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Model architecture
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, 3, activation='relu'),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax'),
])

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10)
]

history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=callbacks)

score = model.evaluate(test_dataset)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

my_number = random.randint(0, 9999999)
model.save(f"test-{my_number}.keras")
print(f"saved as \"test-{my_number}.keras\" ")
