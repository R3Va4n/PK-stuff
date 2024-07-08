import tensorflow as tf

batch_size = 32
epochs = 20

# Manage the Data
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,  # rotate images randomly by up to 10 degrees
    horizontal_flip=True,  # flip images horizontally
)

# Create a generator
augmented_iterator = datagen.flow(train_images, train_labels, batch_size=batch_size)

# Convert the generator to a tf.data.Dataset
augmented_dataset = tf.data.Dataset.from_generator(
    lambda: augmented_iterator,
    output_signature=(
        tf.TensorSpec(shape=(None, 28, 28, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int32)
    )
)

# Ensure the dataset repeats itself enough times
augmented_dataset = augmented_dataset.repeat().batch(batch_size)

# Make the model
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

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
steps_per_epoch = len(train_images) // batch_size
history = model.fit(augmented_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

model.summary()
model.save('my_model.h5')

# Test the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Predictions
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

predictions = probability_model.predict(test_images)
