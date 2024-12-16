import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QScrollArea, QHBoxLayout, QComboBox
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt
from math import ceil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Global variables
selected_images = []  # List to store tuples of (image_path, score)
hiscore = 0  # Global variable to store the high score
current_language = 'en'  # Default language

# Language dictionary
translations = {
    'en': {
        'title': 'NUKI',
        'start_game': 'Start Game',
        'hi_score': 'HI: ',
        'language': 'Select Language',
        'not_correct_image': 'Nuki blinked, this couldn\'t be correct',
        'error_processing_image': 'Error processing image: ',
        'first_screen': "Once upon a time, there was a small, adorable guinea pig named Nuki.<br>Despite being visually impaired, Nuki's heart was full of curiosity and determination.<br>Every morning, he followed the same routine, believing it would set the tone for a wonderful day.<br>At precisely 7 a.m., Nuki would rise from his cozy bed and make his way to the bathroom.<br>There, he would gaze into the mirror, eager to catch a glimpse of his reflection.<br>He looked and thought: How do I look today?",
        'second_screen': "After washing up, he would always step out of his house and greet his roommate Polly.<br> Polly was a clever African Grey parrot and always greeted him just before he could start.<br> 'Good morning, Nuki'",
        'third_screen': "'At least my tail isn't on fire,' she replied.<br> 'Look out the window, that warplane outside is probably not having such a nice day.'",
        'return_to_start': 'Return to Start',
        'next_button_first_screen': 'pretty good, actually',
        'next_button_second_screen': 'Good morning, Polly, you look a bit disheveled today.',
        'next_button_third_screen': "well, that's right"
    },
    'de': {
        'title': 'NUKI',
        'start_game': 'Spiel starten',
        'hi_score': 'HI: ',
        'language': 'Sprache auswählen',
        'not_correct_image': "Nuki blinzelte: Er musste sich verschaut haben",
        'error_processing_image': 'Fehler bei der Bildverarbeitung: ',
        'first_screen': "Es war einmal ein süßes kleines Meerschweinchen namens Nuki.<br> Obwohl es etwas kurtzsichtig war, war sein Herz voller Neugier und Entschlossenheit.<br> Jeden Morgen folgte es der gleichen Routine um so gut wie möglich in den Tag zu starten:.<br> Um genau 7 Uhr morgens erhob sich Nuki voller Tatendrang aus seinem kuscheligen Bett<br> und machte sich auf den Weg ins Badezimmer.<br> Dort schaute er in den Spiegel, begierig darauf, sein Spiegelbild zu erblicken.<br> Er blinzelte und dachte: Wie sehe ich denn heute aus?",
        'second_screen': "Nachdem er sich gewaschen hatte trat er dann immer aus seinem Häuschen und begrüßte seine Mitbewohnerin Polly.<br> Polly war ein schlauer Graupapagei und begrüßte ihn immer kurz bevor er anfangen konnte.<br> \"Guten Morgen Nuki\"<br> \"Guten Morgen Polly, du siehst heute...\"",
        'third_screen': "\"Immerhin brennt mein Schweif nicht\" meinte sie daraufhin.<br> \"Schau mal aus dem Fenster, das Kampfflugzeug da draussen hat glaube ich keinen so schönen Tag.\"",
        'return_to_start': 'Zurück zum Start',
        'next_button_first_screen': 'tatsächlich ziemlich gut',
        'next_button_second_screen': "ein bisschen zerzaust aus.",
        'next_button_third_screen': "\"Das stimmt auf jeden Fall\""
    }
}

def run_cnn(model_path, img_path):
    print(model_path, img_path)
    """
    Load a Keras model from the specified path, preprocess the image,
    and return the predicted class index and its probability.

    Parameters:
    - model_path: str, path to the Keras model file.
    - img_path: str, path to the image file.

    Returns:
    - predicted_index: int, index of the predicted class.
    - probability: float, probability of the predicted class.
    """

    # Load the model
    model = load_model(model_path)

    # Get the input shape from the model
    input_shape = model.input_shape[1:]  # Exclude the batch size dimension

    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    processed_image = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape

    # Make predictions
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions)
    probability = round(100 * (np.max(predictions)), 2)  # Convert to percentage

    return predicted_index, probability

class BaseScreen(QWidget):
    def __init__(self, title, cnn_index, cnn_model_path):
        super().__init__()
        self.title = title
        self.cnn_index = cnn_index
        self.cnn_model_path = cnn_model_path
        self.init_ui()
        self.my_score = 0

    def init_ui(self):
        self.setWindowTitle("Nuki")  # Set the window title to "Nuki"
        self.setGeometry(600, 400, 700, 400)  # Set the geometry of the window
        layout = QVBoxLayout()

        # Common label
        self.label = QLabel(self.title, self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Common button
        self.button = QPushButton('Next', self)
        self.button.clicked.connect(self.open_image_selection_and_process)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def show_next_screen(self):
        self.hide()
        if hasattr(self, 'next_screen'):
            self.next_screen.show()

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select an Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)", options=options)
        return file_name

    def open_image_selection_and_process(self):
        file_name = self.open_file_dialog()
        if file_name:
            try:
                predicted_class_index, accuracy = run_cnn(self.cnn_model_path, file_name)
                if predicted_class_index == self.cnn_index:
                    self.my_score = ceil(accuracy)
                    selected_images.append((file_name, self.my_score))
                    self.show_next_screen()
                else:
                    self.label.setText(translations[current_language]['not_correct_image'])
            except Exception as e:
                self.label.setText(f"{translations[current_language]['error_processing_image']}{str(e)}")

class StartScreen(QWidget):
    def __init__(self, next_screen=None):
        super().__init__()
        self.title = "Start Screen"
        self.next_screen = next_screen  # Define the next screen
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Nuki")  # Set the window title to "Nuki"
        self.setGeometry(600, 400, 700, 400)  # Set the geometry of the window
        layout = QVBoxLayout()

        # Title label
        self.label = QLabel(translations[current_language]['title'], self)
        self.label.setAlignment(Qt.AlignCenter)
        font = QFont('Arial', 64)  # Set font to Arial, size 64
        self.label.setFont(font)
        layout.addWidget(self.label)

        # High score label
        self.hiscore_label = QLabel(f"{translations[current_language]['hi_score']}{hiscore}", self)
        self.hiscore_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.hiscore_label)  # Add high score label above the button

        # Language selection dropdown
        self.language_dropdown = QComboBox(self)
        self.language_dropdown.addItems(['English', 'Deutsch'])  # Add language options
        self.language_dropdown.currentIndexChanged.connect(self.change_language)  # Connect change event
        layout.addWidget(self.language_dropdown)

        # Start Game button
        self.button = QPushButton(translations[current_language]['start_game'], self)
        self.button.clicked.connect(self.start_game)
        layout.addWidget(self.button)  # Button is added after the high score label

        self.setLayout(layout)

    def start_game(self):
        self.hide()  # Hide the start screen
        if self.next_screen:
            self.next_screen.show()  # Show the next screen

    def change_language(self):
        # Change the current language based on the dropdown selection
        global current_language
        if self.language_dropdown.currentIndex() == 0:  # English selected
            current_language = 'en'
        else:  # German selected
            current_language = 'de'

        # Update UI text based on the selected language
        self.label.setText(translations[current_language]['title'])
        self.hiscore_label.setText(f"{translations[current_language]['hi_score']}{hiscore}")
        self.button.setText(translations[current_language]['start_game'])

        # Update all screens
        if self.next_screen:
            self.next_screen.update_text()

class FirstScreen(BaseScreen):
    def __init__(self, next_screen=None):
        super().__init__("First Screen", 1, "NUKI test-7513932.keras")
        self.next_screen = next_screen  # Define the next screen
        self.update_text()

    def update_text(self):
        self.label.setText(translations[current_language]['first_screen'])
        self.button.setText(translations[current_language]['next_button_first_screen'])
        self.next_screen.update_text()

class SecondScreen(BaseScreen):
    def __init__(self, next_screen=None):
        super().__init__("Second Screen", 0, "NUKI test-7513932.keras")
        self.next_screen = next_screen  # Define the next screen
        self.update_text()

    def update_text(self):
        self.label.setText(translations[current_language]['second_screen'])
        self.button.setText(translations[current_language]['next_button_second_screen'])
        self.next_screen.update_text()

class ThirdScreen(BaseScreen):
    def __init__(self, next_screen=None):
        super().__init__("Third Screen", 2, "NUKI test-7513932.keras")
        self.next_screen = next_screen  # Define the next screen
        self.update_text()

    def update_text(self):
        self.label.setText(translations[current_language]['third_screen'])
        self.button.setText(translations[current_language]['next_button_third_screen'])
        self.next_screen.update_text()

class EndScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("End Screen")  # Set the window title
        self.setGeometry(600, 400, 700, 400)  # Set the geometry of the window
        self.layout = QVBoxLayout()  # Create a layout for the EndScreen
        self.setLayout(self.layout)  # Set the layout

        # Add a button to return to the start screen
        self.return_button = QPushButton(translations[current_language]['return_to_start'], self)
        self.return_button.clicked.connect(self.return_to_start)
        self.layout.addWidget(self.return_button)

    def display_selected_images(self):
        # Clear the layout before adding new images
        for i in reversed(range(self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget is not None and widget != self.return_button:  # Keep the return button
                widget.deleteLater()

        # Calculate the total score from selected images
        total_score = sum(score for _, score in selected_images)
        global hiscore

        # Update the high score if the current score exceeds it
        if total_score > hiscore:
            hiscore = total_score

        # Create the label with the current score and high score
        label = QLabel(f"Score: {total_score}   HI: {hiscore}", self)  # Set label for score and high score
        label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(label)

        # Create a scroll area to display images
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        for image_path, score in selected_images:  # Iterate through the global list of tuples
            # Create a horizontal layout for each image and its label
            h_layout = QHBoxLayout()
            pixmap = QPixmap(image_path)

            # Maintain aspect ratio while scaling
            scaled_pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label = QLabel(self)
            image_label.setPixmap(scaled_pixmap)
            h_layout.addWidget(image_label)

            # Create a label for the image path and score
            path_label = QLabel(f"{image_path.split('/')[-1]} (Score: {score}/100)", self)  # Display only the file name and score
            h_layout.addWidget(path_label)

            scroll_layout.addLayout(h_layout)  # Add the horizontal layout to the scroll layout

        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        self.layout.addWidget(scroll_area)

    def return_to_start(self):
        global selected_images
        selected_images.clear()  # Clear the selected images when returning to start
        self.hide()
        start_screen.show()  # Show the start screen again

    def show(self):
        super().show()  # Call the base class show method
        self.display_selected_images()  # Update the displayed images when the screen is shown

    def update_text(self):
        self.return_button.setText(translations[current_language]['return_to_start'])

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Create screen instances
    end_screen = EndScreen()
    third_screen = ThirdScreen(next_screen=end_screen)
    second_screen = SecondScreen(next_screen=third_screen)
    first_screen = FirstScreen(next_screen=second_screen)
    start_screen = StartScreen(next_screen=first_screen)

    # Start the application
    start_screen.show()
    sys.exit(app.exec_())
