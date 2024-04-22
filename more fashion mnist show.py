import sys
from random import randint
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox
import tensorflow as tf


class MainClass(QWidget):

    def __init__(self):
        super().__init__()
        self.current_pic_index = None
        self.init_ui()

    def init_ui(self):
        # Set up the main window
        self.setWindowTitle('FILE TITLE')
        self.setGeometry(100, 100, 800, 600)
        self.current_pic_index = 0
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                            'Ankle boot']

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()

        # Create a Matplotlib figure
        self.my_figure = plt.figure()
        plt.imshow(self.test_images[self.current_pic_index], cmap=plt.cm.binary)

        # Create widgets
        self.my_canvas = FigureCanvas(self.my_figure)

        # buttons
        random_picture_button = QPushButton('New Random Picture', self)
        right_picture_button = QPushButton(">", self)
        left_picture_button = QPushButton("<", self)

        # labels
        self.index_label = QLabel(self)
        self.label_label = QLabel(self)

        # spinboxes
        self.index_spinbox = QSpinBox(self)
        self.index_spinbox.setMaximum(len(self.test_images)-1)

        # setup labels and spinboxes
        self.refresh_labels()

        # Connect button click
        random_picture_button.clicked.connect(self.on_random_picture_button_click)
        right_picture_button.clicked.connect(self.on_right_picture_button_click)
        left_picture_button.clicked.connect(self.on_left_picture_button_click)

        # connect spinbox change
        self.index_spinbox.valueChanged.connect(self.on_spinbox_change)


        # Create a layout and add widgets to it
        layout = QVBoxLayout(self)
        layout.addWidget(self.my_canvas)

        index_layout = QHBoxLayout(self)
        index_layout.addWidget(self.index_label)
        index_layout.addWidget(self.index_spinbox)
        layout.addLayout(index_layout)
        layout.addWidget(self.label_label)

        button_layout = QHBoxLayout(self)
        button_layout.addWidget(random_picture_button)
        button_layout.addWidget(left_picture_button)
        button_layout.addWidget(right_picture_button)
        layout.addLayout(button_layout)

        # Set the layout for the main window
        self.setLayout(layout)

        # Show the main window
        self.show()

    def on_random_picture_button_click(self):
        self.current_pic_index = randint(0, len(self.test_images) - 1)
        self.refresh_plot()
        self.refresh_labels()

    def on_right_picture_button_click(self):
        self.current_pic_index += 1
        if self.current_pic_index >= len(self.test_images):
            self.current_pic_index = 0
        self.refresh_plot()
        self.refresh_labels()

    def on_left_picture_button_click(self):
        self.current_pic_index -= 1
        if self.current_pic_index <= 0:
            self.current_pic_index = len(self.test_images) - 1
        self.refresh_plot()
        self.refresh_labels()

    def on_spinbox_change(self):
        self.current_pic_index = self.index_spinbox.value()
        self.refresh_plot()
        self.refresh_labels()

    def refresh_plot(self):
        plt.clf()
        plt.imshow(self.test_images[self.current_pic_index], cmap=plt.cm.binary)
        plt.gca().set_aspect('auto')
        self.my_canvas.draw()

    def refresh_labels(self):
        self.index_label.setText("Index:")
        self.index_spinbox.setValue(self.current_pic_index)
        self.label_label.setText(f"Label: {self.class_names[self.test_labels[self.current_pic_index]]}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainClass()
    sys.exit(app.exec_())
