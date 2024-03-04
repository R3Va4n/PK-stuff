import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QCheckBox

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
        # Connect button click
        button.clicked.connect(self.onButtonClick)

        # Create a vertical layout and add widgets to it
        layout = QVBoxLayout(self)
        layout.addWidget(self.checkbox)
        layout.addWidget(button)

        # Set the layout for the main window
        self.setLayout(layout)

        # Show the main window
        self.show()

    def onButtonClick(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", options=options)
        if directory:
            print(f"Selected Directory: {directory}")
            only_images = self.checkbox.isChecked()
            self.print_dir(directory, only_images)

    def print_dir(self, dir_path, only_images=False):
        image_formats = (".jpg", ".jpeg", ".ico", ".png", ".PNG", ".webp")
        for i in os.listdir(dir_path):
            if not only_images or os.path.splitext(i)[1] in image_formats:
                print(i)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileViewer5000()
    sys.exit(app.exec_())
