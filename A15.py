import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox, QLineEdit
from random import randint

class Main_Class(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # Set up the main window
        self.setWindowTitle('A15')
        self.setGeometry(100, 100, 300, 200)

        # Create widgets
        # buttons
        exit_button = QPushButton('Exit', self)
        generate_random_numbers_button = QPushButton('Generate', self)
        generate_square_numbers_button = QPushButton('Generate', self)
        # labels
        random_number_label = QLabel(self)
        random_number_label.setText("random number generator")
        square_number_label = QLabel(self)
        square_number_label.setText("square number generator")
        # SpinBoxes
        self.random_number_1_Spinbox = QSpinBox(self)
        self.random_number_2_Spinbox = QSpinBox(self)
        self.square_numbers_Spinbox = QSpinBox(self)
        self.random_number_1_Spinbox.setMaximum(int(99999))
        self.random_number_2_Spinbox.setMaximum(int(99999))
        self.square_numbers_Spinbox.setMaximum(int(99999))
        # LineEdits
        self.random_numbers_out = QLineEdit(self)
        self.square_numbers_out = QLineEdit(self)
        self.random_numbers_out.setReadOnly(True)
        self.square_numbers_out.setReadOnly(True)

        # Connect button click
        exit_button.clicked.connect(self.on_button_click_exit)
        generate_random_numbers_button.clicked.connect(self.on_button_random_numbers)
        generate_square_numbers_button.clicked.connect(self.on_button_square_numbers)
        # Create a layout and add widgets to it
        layout = QVBoxLayout(self)
        layout.addWidget(random_number_label)

        random_numbers_box = QHBoxLayout(self)
        spinbox_box = QVBoxLayout(self)
        spinbox_box.addWidget(self.random_number_1_Spinbox)
        spinbox_box.addWidget(self.random_number_2_Spinbox)
        random_numbers_box.addLayout(spinbox_box)
        random_numbers_box.addWidget(generate_random_numbers_button)
        random_numbers_box.addWidget(self.random_numbers_out)
        layout.addLayout(random_numbers_box)

        layout.addWidget(square_number_label)

        square_numbers_box = QHBoxLayout(self)
        square_numbers_box.addWidget(self.square_numbers_Spinbox)
        square_numbers_box.addWidget(generate_square_numbers_button)
        square_numbers_box.addWidget(self.square_numbers_out)
        layout.addLayout(square_numbers_box)

        layout.addWidget(exit_button)

        # Set the layout for the main window
        self.setLayout(layout)

        # Show the main window
        self.show()

    def on_button_click_exit(self) -> None:
        self.beende()

    def on_button_random_numbers(self) -> None:
        max_nr = max(self.random_number_1_Spinbox.value(), self.random_number_2_Spinbox.value())
        min_nr = min(self.random_number_1_Spinbox.value(), self.random_number_2_Spinbox.value())
        print(f"min:{min_nr} max:{max_nr}")
        random_number = self.macheZufall(min_nr, max_nr)
        print(f"random_number:{random_number}")
        self.random_numbers_out.setText(str(random_number))

    def on_button_square_numbers(self) -> None:
        number = self.sumGeradeQuadrat(self.square_numbers_Spinbox.value())
        self.square_numbers_out.setText(str(number))

    def beende(self) -> None:
        sys.exit()

    def macheZufall(self, a, b) -> int:
        return randint(a, b)

    def sumGeradeQuadrat(self, n):
        total = 0
        for i in range(n):
            total += (i*2)*(i*2)    # I know that it is (i*i + 4i+ 4), idc
            print(f"total is now {total}, after adding the square of {i*2}")
        return total

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Main_Class()
    sys.exit(app.exec_())
