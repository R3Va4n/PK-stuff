import sys
import json
import os
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QResizeEvent
from PyQt5.QtCore import Qt
from tensorflow.keras.models import load_model
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ImageLabel(QLabel):
    """Custom QLabel that maintains aspect ratio when scaling images."""

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignCenter)
        self._pixmap = None

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        super().setPixmap(self._scale_pixmap())

    def _scale_pixmap(self):
        if self._pixmap and not self._pixmap.isNull():
            return self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return self._pixmap

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap:
            super().setPixmap(self._scale_pixmap())


class CNNViewer(QWidget):
    """Main application window for CNN model visualization."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.class_labels = None
        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        self.setup_ui()

    def setup_ui(self):
        """Initialize the user interface."""
        self.setGeometry(100, 100, 800, 400)
        main_layout = QHBoxLayout()

        # Left side - Chart
        chart_layout = QVBoxLayout()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_layout.addWidget(self.canvas)

        # Right side - Controls and image
        right_layout = self.create_right_layout()

        # Combine layouts
        main_layout.addLayout(chart_layout)
        main_layout.addLayout(right_layout)
        main_layout.setStretch(0, 1)
        main_layout.setStretch(1, 1)
        self.setLayout(main_layout)

    def create_right_layout(self):
        """Create the right side layout with controls and image display."""
        layout = QVBoxLayout()

        # Control buttons
        button_layout = QHBoxLayout()
        self.create_buttons(button_layout)
        layout.addLayout(button_layout)

        # Image display
        self.image_label = ImageLabel()
        layout.addWidget(self.image_label)

        # Status text
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        return layout

    def create_buttons(self, layout):
        """Create and configure control buttons."""
        buttons = {
            'Load CNN Model': self.load_model,
            'Load Labels': self.load_labels,
            'Choose Image': self.load_image,
            'Run CNN': self.run_cnn
        }

        for text, callback in buttons.items():
            button = QPushButton(text)
            button.clicked.connect(callback)
            layout.addWidget(button)

    def load_model(self):
        """Load a CNN model from file."""
        filename, _ = QFileDialog.getOpenFileName(self, "Load CNN Model", "", "Keras Model Files (*.keras *.h5)")
        if not filename:
            return

        try:
            self.model = load_model(filename)
            self.try_load_labels(filename)
            self.update_status(f'Model loaded from: {filename}')
            self.run_cnn()
        except Exception as e:
            self.update_status(f'Error loading model: {str(e)}')

    def try_load_labels(self, model_path):
        """Attempt to load labels from accompanying JSON file."""
        label_file = Path(model_path).with_suffix('.json')
        if label_file.exists():
            try:
                self.class_labels = json.loads(label_file.read_text())
                self.update_status(f'Labels loaded automatically')
            except Exception:
                pass

    def load_labels(self):
        """Load class labels from a JSON file."""
        filename, _ = QFileDialog.getOpenFileName(self, "Load Labels File", "", "JSON Files (*.json)")
        if not filename:
            return

        try:
            with open(filename) as f:
                self.class_labels = json.load(f)
            self.update_status(f'Labels loaded from: {filename}')
            self.run_cnn()
        except Exception as e:
            self.update_status(f'Error loading labels: {str(e)}')

    def load_image(self):
        """Load an image file for classification."""
        filename, _ = QFileDialog.getOpenFileName(self, "Choose Image", "",
                                                  "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if filename:
            self.image_label.setPixmap(QPixmap(filename))
            self.run_cnn()

    def update_chart(self, probabilities, labels):
        """Update the probability chart."""
        self.figure.clear()

        # Filter probabilities above threshold
        threshold = 0.05
        mask = probabilities > threshold
        probs = probabilities[mask]
        labels = [labels[i] for i in range(len(labels)) if mask[i]]

        # Create chart
        ax = self.figure.add_subplot(111)
        bars = ax.bar(range(len(probs)), probs)

        # Configure chart
        ax.set_xticks(range(len(probs)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Probability')
        ax.set_title(f'Class Probabilities > {threshold}')
        ax.set_ylim(0, 1)
        ax.yaxis.grid(True, linestyle='--', which='major', alpha=0.7)
        ax.set_yticks(np.arange(0, 1.1, 0.1))

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.2f}', ha='center', va='bottom')

        # Finalize chart
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        self.figure.tight_layout()
        self.canvas.draw()

    def run_cnn(self):
        """Run the CNN model on the current image."""
        if not self.model:
            self.update_status("Please load a CNN model first.")
            return

        if not self.image_label._pixmap:
            self.update_status("No image loaded.")
            return

        try:
            # Prepare image
            image = self.prepare_image()

            # Run prediction
            output = self.model.predict(image)
            class_index = np.argmax(output[0])
            probability = output[0, class_index]

            # Get class label
            class_label = (self.class_labels.get(str(class_index), "Unknown Class")
                           if self.class_labels else f"Class {class_index}")

            # Update UI
            self.update_status(f'Class: {class_label} (Index: {class_index}), '
                               f'Probability: {probability:.2f}')

            # Update chart
            labels = ([self.class_labels.get(str(i), f"Class {i}")
                       for i in range(len(output[0]))] if self.class_labels
                      else [f"Class {i}" for i in range(len(output[0]))])
            self.update_chart(output[0], labels)

        except Exception as e:
            self.update_status(f'Error running CNN: {str(e)}')

    def prepare_image(self):
        """Prepare image for CNN processing."""
        image = self.image_label._pixmap.toImage()

        # Convert to numpy array
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)[:, :, :3]

        # Preprocess
        arr = arr / 255.0
        input_shape = self.model.input_shape[1:3]
        arr = np.resize(arr, (*input_shape, 3))
        return np.expand_dims(arr, axis=0)

    def update_status(self, message):
        """Update status message."""
        self.status_label.setText(message)

    def resizeEvent(self, event: QResizeEvent):
        """Handle window resize events."""
        super().resizeEvent(event)
        self.figure.set_size_inches(self.canvas.width() / self.figure.dpi,
                                    self.canvas.height() / self.figure.dpi)
        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CNNViewer()
    window.show()
    sys.exit(app.exec_())