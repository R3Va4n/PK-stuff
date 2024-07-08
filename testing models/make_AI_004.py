import tensorflow as tf
import numpy as np

batch_size = 16
epochs = 30

# Manage the Data
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,  # rotate images randomly by up to 10 degrees
    horizontal_flip=True,  # flip images horizontally
)

augmented_dataset = datagen.flow(train_images, train_labels, batch_size=batch_size)

# make the model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1.255),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(10)
])
# train the model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(augmented_dataset, epochs=epochs, steps_per_epoch=len(train_images) // batch_size)

model.summary()
model.save('my_model.h5')

# test the model
probability_model = tf.keras.Sequential([
    tf.keras.layers.Softmax()
])

predictions = probability_model.predict(test_images)
loss, accuracy = model.evaluate(test_images, test_labels)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
