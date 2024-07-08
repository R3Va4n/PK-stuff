import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QDateTimeEdit, \
    QPlainTextEdit, QFileDialog


class Main_Class(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Set up the main window
        self.setWindowTitle('text editor')
        self.setGeometry(100, 100, 1000, 600)

        # Create widgets
        save_button = QPushButton('Save', self)
        load_button = QPushButton('Load', self)
        datetime_widget = QDateTimeEdit(self)
        self.plain_text_edit = QPlainTextEdit(self)

        # Connect button click
        save_button.clicked.connect(self.on_button_click_save)
        load_button.clicked.connect(self.on_button_click_load)

        # Create a vertical layout and add widgets to it
        layout = QVBoxLayout(self)
        layout_h = QHBoxLayout(self)
        layout_h.addWidget(load_button)
        layout_h.addWidget(save_button)
        layout.addLayout(layout_h)
        layout.addWidget(datetime_widget)
        layout.addWidget(self.plain_text_edit)

        # Set the layout for the main window
        self.setLayout(layout)

        # Show the main window
        self.show()

    def on_button_click_load(self) -> None:
        options = QFileDialog.Options()
        file_path = QFileDialog.getOpenFileName()[0]
        text = open(file_path, "r", encoding="utf-8").read()
        self.plain_text_edit.insertPlainText(text)

    def on_button_click_save(self) -> None:
        print("hello")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main_Class()
    sys.exit(app.exec_())
