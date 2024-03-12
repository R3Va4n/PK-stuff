import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QCheckBox, QLabel

class FileViewer5000(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Set up the main window
        self.setWindowTitle('FileViewer5000')
        self.setGeometry(100, 100, 800, 600)

        # Create widgets
        button = QPushButton('Print items', self)
        self.checkbox = QCheckBox('Only pictures', self)
        self.picture_number_label = QLabel(self)

        # Connect button click
        button.clicked.connect(self.onButtonClick)

        # Create a vertical layout and add widgets to it
        layout = QVBoxLayout(self)
        layout.addWidget(self.checkbox)
        layout.addWidget(self.picture_number_label)
        layout.addWidget(button)

        # Set the layout for the main window
        self.setLayout(layout)

        # Show the main window
        self.show()

    def onButtonClick(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            print(f"Selected Directory: {directory}")
            only_images = self.checkbox.isChecked()
            self.print_dir(directory, only_images)

    def print_dir(self, dir_path, only_images=False):
        image_formats = (".jpg", ".jpeg", ".ico", ".png", ".PNG", ".webp")
        number_images = 0
        for i in os.listdir(dir_path):
            if (not only_images) or (os.path.splitext(i)[1] in image_formats):
                print(i)
                if (os.path.splitext(i)[1] in image_formats):
                    number_images += 1
        self.picture_number_label.setText(f"Total number of images: {number_images} ")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileViewer5000()
    sys.exit(app.exec_())
