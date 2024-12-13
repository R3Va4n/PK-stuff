import io
import random
import json  # Import the json module
import time  # Import the time module

import numpy as np
import tensorflow as tf
import keras
from keras import layers

import matplotlib.pyplot as plt

# Model / data parameters
num_classes = 3
input_shape = (128, 128, 3)
batch_size = 32
epochs = 25
augmentation_strength = 0.25  # 0.5 #1 #2
train_dataset_path = 'images/128x128/train'
test_dataset_path = 'images/128x128/test'
activation_function = 'swish'

# Load the data and split it between train and test sets
train_dataset = keras.utils.image_dataset_from_directory(
    directory=train_dataset_path,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(input_shape[0], input_shape[1]),
    shuffle=True  # Shuffle the training dataset
)

# Retrieve class names and create a mapping of class indices
class_names = train_dataset.class_names
class_indices = {name: index for index, name in enumerate(class_names)}

test_dataset = keras.utils.image_dataset_from_directory(
    directory=test_dataset_path,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(input_shape[0], input_shape[1]),
    shuffle=False  # Keep test dataset deterministic for evaluation
)

# Data normalization
normalization_layer = layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1 * augmentation_strength),
    layers.RandomZoom(0.1 * augmentation_strength),
    layers.RandomTranslation(height_factor=0.1 * augmentation_strength, width_factor=0.1 * augmentation_strength)
])
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Prefetching data to increase speed
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Model architecture
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, 3, activation=activation_function),
    layers.BatchNormalization(),
    layers.Conv2D(32, 3, activation=activation_function),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, 3, activation=activation_function),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, activation=activation_function),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, 3, activation=activation_function),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, activation=activation_function),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation=activation_function),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax'),
])

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacks = [
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=7),
]

# Start timing the training
start_time = time.time()

history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, callbacks=callbacks)

# End timing the training
training_time = time.time() - start_time
print(f"Model training time: {training_time:.2f} seconds")

score = model.evaluate(test_dataset)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

my_number = random.randint(0, 9999999)

# Save the model
model.save(f"test-{my_number}.h5")
print(f"saved as \"test-{my_number}.h5\" ")

# Save class indices to a JSON file in the correct format
correct_class_indices = {str(index): name for index, name in enumerate(class_names)}
with open(f"test-{my_number}.json", 'w') as json_file:
    json.dump(correct_class_indices, json_file)

print(f"Class indices saved as \"test-{my_number}_labels.json\" ")

# Plot the training and validation accuracy and loss
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Training Loss')
plt.legend()
plt.show()
