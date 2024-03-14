import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QLineEdit


def calculate_digit_proportion(file_path) -> float:
    my_file_text = open(file_path, "r")
    number_digits = 0
    total_characters = 0
    for char in my_file_text:
        if char.isdigit():
            number_digits += 1
        total_characters += 1

    if total_characters == 0:
        total_characters = 1

    return number_digits/total_characters

def reduce_empty_lines(file_path):
    file_text = open(file_path, "r").read()

    # Remove empty lines
    lines = file_text.split('\n')
    non_empty_lines = []
    for line in lines:
        if line.strip() != "":
            non_empty_lines.append(line)

    # Construct the new file
    reduced_text = '\n'.join(non_empty_lines)
    file_name_parts = file_path.split('.')
    new_file_path = f"{file_name_parts[0]}.reduziert.{file_name_parts[-1]}"
    new_file = open(new_file_path, 'w')
    new_file.write(reduced_text)
    print(f"File reduced, saved to {new_file_path}")

class MainClass(QWidget):

    def __init__(self):
        super().__init__()
        self.file_path = None
        self.digit_proportion_LineEdit = None
        self.current_file_text = None
        self.current_file = None
        self.init_ui()
        self.digit_proportion = 0

    def init_ui(self):
        # Set up wierd stuff
        self.digit_proportion = 0

        # Set up the main window
        self.setWindowTitle('A15')
        self.setGeometry(100, 100, 420, 420)

        # Create widgets

        # LineEdits
        self.digit_proportion_LineEdit = QLineEdit(self)
        self.digit_proportion_LineEdit.setReadOnly(True)
        self.update_digit_proportion_lineedit()

        # buttons
        load_button = QPushButton('Load File', self)
        save_button = QPushButton('Save reduced File', self)

        # Connect button click
        load_button.clicked.connect(self.on_button_click_load)
        save_button.clicked.connect(self.on_button_click_save_reduced)

        # Create a layout and add widgets to it
        layout = QVBoxLayout(self)
        layout.addWidget(load_button)
        layout.addWidget(save_button)
        layout.addWidget(self.digit_proportion_LineEdit)

        # Set the layout for the main window
        self.setLayout(layout)

        # Show the main window
        self.show()

    def on_button_click_load(self) -> None:
        self.file_path = QFileDialog.getOpenFileName()[0]
        self.current_file = open(self.file_path, "r", encoding="utf-8")
        self.current_file_text = self.current_file.read()
        self.digit_proportion = calculate_digit_proportion(self.file_path)
        self.update_digit_proportion_lineedit()

    def on_button_click_save_reduced(self) -> None:
        reduce_empty_lines(self.file_path)

    def update_digit_proportion_lineedit(self) -> None:
        self.digit_proportion_LineEdit.setText(f"proportion of digits: {self.digit_proportion}%")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainClass()
    sys.exit(app.exec_())
