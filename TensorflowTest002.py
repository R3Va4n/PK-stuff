import tensorflow as tf
import numpy as np

batch_size = 16
epochs = 20

# Manage the Data
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def random_occlusion(image):
    # settings - the current values are from the Paper about random occlusion
    erasure_probability = 0.5
    area_ratio_erasing_region_min = 0.02
    area_ratio_erasing_region_max = 0.4
    erasing_region_aspect_ratio_min = 0.3
    erasing_region_aspect_ratio_max = 1/erasing_region_aspect_ratio_min

    # Copy the input image to avoid modifying the original image
    occluded_image = np.copy(image)

    # Get the dimensions of the image
    height, width, _ = occluded_image.shape
    area = height * width

    if np.random.default_rng().random() > erasure_probability:
        return occluded_image
    else:
        while True:
            area_ratio_erasing_region = np.random.uniform(area_ratio_erasing_region_min, area_ratio_erasing_region_max) * area
            erasing_region_aspect_ratio = np.random.uniform(erasing_region_aspect_ratio_min, erasing_region_aspect_ratio_max)
            erasing_region_height = np.sqrt(area_ratio_erasing_region * erasing_region_aspect_ratio)
            erasing_region_width = np.sqrt(area_ratio_erasing_region/erasing_region_aspect_ratio)
            erasing_region_position_x = np.random.uniform(0, width)
            erasing_region_position_y = np.random.uniform(0, height)
            if erasing_region_position_x + erasing_region_width <= width and erasing_region_position_y+erasing_region_height <= height:
                for pixel_x in range(int(erasing_region_width)):
                    for pixel_y in range(int(erasing_region_height)):
                        occluded_image[pixel_x][pixel_y] = np.random.randint(0, 256)
                return occluded_image


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,  # rotate images randomly by up to 10 degrees
    horizontal_flip=True,  # flip images horizontally
    preprocessing_function=random_occlusion
)

augmented_dataset = datagen.flow(train_images, train_labels, batch_size=batch_size)

"""# Create a dataset from the training images and labels
augmented_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
augmented_dataset = augmented_dataset.shuffle(buffer_size=len(train_images)).batch(batch_size).repeat()"""


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
