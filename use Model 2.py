import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap
from tensorflow.keras.models import load_model
import numpy as np


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None  # Initialize model variable
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        self.button_load_model = QPushButton('Load CNN Model')
        self.button_load_model.clicked.connect(self.load_cnn)
        layout.addWidget(self.button_load_model)

        self.button_choose_image = QPushButton('Choose Image')
        self.button_choose_image.clicked.connect(self.open_image)
        layout.addWidget(self.button_choose_image)

        self.label = QLabel()
        layout.addWidget(self.label)

        self.button_run_cnn = QPushButton('Run CNN')
        self.button_run_cnn.clicked.connect(self.run_cnn)
        layout.addWidget(self.button_run_cnn)

        self.text_label = QLabel()
        layout.addWidget(self.text_label)

        self.setLayout(layout)

    def load_cnn(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load CNN Model", "", "Keras Model Files (*.keras)")
        if filename:
            self.model = load_model(filename)
            self.text_label.setText(f'Model loaded from: {filename}')

    def open_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if filename:
            pixmap = QPixmap(filename)
            self.label.setPixmap(pixmap)

    def run_cnn(self):
        if self.model is None:
            self.text_label.setText("Please load a CNN model first.")
            return

        # Get the image from the label
        pixmap = self.label.pixmap()

        if pixmap is None:
            self.text_label.setText("No image loaded.")
            return

        # Convert QPixmap to QImage
        image = pixmap.toImage()

        # Convert QImage to NumPy array
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        image_array = np.array(ptr).reshape((height, width, 4))  # Assuming 4 channels (RGBA)

        # Convert to RGB if necessary
        if image_array.shape[2] == 4:  # If the image has an alpha channel
            image_array = image_array[:, :, :3]  # Discard the alpha channel

        # Preprocess the image
        image_array = image_array / 255.0

        # Get the input shape from the model
        input_shape = self.model.input_shape[1:3]  # Get height and width

        # Resize the image to the input shape of the model
        image_array = np.resize(image_array, (input_shape[0], input_shape[1], 3))

        # Add a batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # Run the CNN
        output = self.model.predict(image_array)

        # Get the class with the highest probability
        class_index = np.argmax(output)
        probability = output[0, class_index]

        # Display the output and probability
        self.text_label.setText(f'Class: {class_index}, Probability: {probability:.2f}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())