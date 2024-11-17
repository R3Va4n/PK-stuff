import sys
import json
import os
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog, QLabel,
    QVBoxLayout, QHBoxLayout, QSizePolicy, QComboBox,
    QScrollArea, QFrame, QGridLayout, QTabWidget
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
        self.selected = False
        self.setStyleSheet("border: 2px solid lightgray;")

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

    def mousePressEvent(self, event):
        self.selected = not self.selected
        self.setStyleSheet(f"border: 2px solid {'blue' if self.selected else 'lightgray'};")
        super().mousePressEvent(event)


class ModelManager:
    """Manages multiple CNN models and their labels."""

    def __init__(self):
        self.models = {}  # {name: (model, labels)}

    def add_model(self, name, model, labels=None):
        self.models[name] = (model, labels)

    def get_model(self, name):
        return self.models.get(name, (None, None))

    def get_model_names(self):
        return list(self.models.keys())

    def remove_model(self, name):
        if name in self.models:
            del self.models[name]


class ImageManager:
    """Manages multiple images for classification."""

    def __init__(self):
        self.images = {}  # {name: QPixmap}

    def add_image(self, name, pixmap):
        self.images[name] = pixmap

    def get_image(self, name):
        return self.images.get(name)

    def get_image_names(self):
        return list(self.images.keys())

    def remove_image(self, name):
        if name in self.images:
            del self.images[name]


class CNNViewer(QWidget):
    """Main application window for CNN model visualization."""

    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.image_manager = ImageManager()
        self.setup_ui()

    def setup_ui(self):
        """Initialize the user interface."""
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowTitle("Multi-Model CNN Viewer")

        # Main layout
        main_layout = QVBoxLayout()

        # Control panel
        control_panel = self.create_control_panel()
        main_layout.addLayout(control_panel)

        # Tab widget for different views
        self.tab_widget = QTabWidget()
        self.setup_tabs()
        main_layout.addWidget(self.tab_widget)

        self.setLayout(main_layout)

    def setup_tabs(self):
        """Create tabs for different views."""
        # Image Gallery Tab
        self.gallery_tab = QWidget()
        gallery_layout = QVBoxLayout()
        self.image_grid = QGridLayout()
        scroll = QScrollArea()
        scroll_content = QWidget()
        scroll_content.setLayout(self.image_grid)
        scroll.setWidget(scroll_content)
        scroll.setWidgetResizable(True)
        gallery_layout.addWidget(scroll)
        self.gallery_tab.setLayout(gallery_layout)
        self.tab_widget.addTab(self.gallery_tab, "Image Gallery")

        # Results Tab
        self.results_tab = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_tab.setLayout(self.results_layout)
        self.tab_widget.addTab(self.results_tab, "Results")

    def create_control_panel(self):
        """Create the control panel with buttons and dropdowns."""
        control_layout = QHBoxLayout()

        # Model controls group
        model_group = QVBoxLayout()
        model_group.addWidget(QLabel("Models:"))

        model_buttons = QHBoxLayout()
        self.add_model_btn = QPushButton("Add Model")
        self.add_model_btn.clicked.connect(self.load_model)
        model_buttons.addWidget(self.add_model_btn)

        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        model_group.addLayout(model_buttons)
        model_group.addWidget(self.model_combo)

        # Image controls group
        image_group = QVBoxLayout()
        image_group.addWidget(QLabel("Images:"))

        image_buttons = QHBoxLayout()
        self.add_image_btn = QPushButton("Add Images")
        self.add_image_btn.clicked.connect(self.load_images)
        image_buttons.addWidget(self.add_image_btn)

        self.clear_images_btn = QPushButton("Clear Images")
        self.clear_images_btn.clicked.connect(self.clear_images)
        image_buttons.addWidget(self.clear_images_btn)

        image_group.addLayout(image_buttons)

        # Run control group
        run_group = QVBoxLayout()
        run_group.addWidget(QLabel("Actions:"))
        self.run_btn = QPushButton("Run Selected Models")
        self.run_btn.clicked.connect(self.run_selected)
        run_group.addWidget(self.run_btn)

        # Add groups to main control layout
        control_layout.addLayout(model_group)
        control_layout.addLayout(image_group)
        control_layout.addLayout(run_group)

        # Create a frame for the control panel
        control_frame = QFrame()
        control_frame.setLayout(control_layout)
        control_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)

        # Wrap in another layout to add some padding
        wrapped_layout = QVBoxLayout()
        wrapped_layout.addWidget(control_frame)

        return wrapped_layout

    def clear_images(self):
        """Clear all images from the grid and image manager."""
        # Clear grid
        for i in reversed(range(self.image_grid.count())):
            widget = self.image_grid.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Clear image manager
        self.image_manager = ImageManager()

        # Update status
        self.show_status("All images cleared")

    def load_model(self):
        """Load a CNN model and its labels."""
        filename, _ = QFileDialog.getOpenFileName(self, "Load CNN Model", "",
                                                  "Keras Model Files (*.keras *.h5)")
        if not filename:
            return

        try:
            model = load_model(filename)
            model_name = Path(filename).stem
            labels = self.try_load_labels(filename)

            self.model_manager.add_model(model_name, model, labels)
            self.update_model_combo()

        except Exception as e:
            self.show_status(f'Error loading model: {str(e)}')

    def load_images(self):
        """Load multiple images for classification."""
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Choose Images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)")

        if not filenames:
            return

        # Clear existing images from grid
        for i in reversed(range(self.image_grid.count())):
            widget = self.image_grid.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        # Add new images
        for row, filename in enumerate(filenames):
            try:
                pixmap = QPixmap(filename)
                image_name = Path(filename).stem

                # Store image in manager
                self.image_manager.add_image(image_name, pixmap)

                # Create and configure image label
                label = ImageLabel()
                label.setPixmap(pixmap)

                # Add to grid (3 columns)
                self.image_grid.addWidget(label, row // 3, row % 3)

            except Exception as e:
                self.show_status(f'Error loading image {filename}: {str(e)}')

    def try_load_labels(self, model_path):
        """Attempt to load labels from accompanying JSON file."""
        label_file = Path(model_path).with_suffix('.json')
        if label_file.exists():
            try:
                return json.loads(label_file.read_text())
            except Exception:
                pass
        return None

    def update_model_combo(self):
        """Update the model selection combo box."""
        self.model_combo.clear()
        self.model_combo.addItems(self.model_manager.get_model_names())

    def prepare_image(self, pixmap, input_shape):
        """Prepare image for CNN processing."""
        image = pixmap.toImage()

        # Convert to numpy array
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(image.height(), image.width(), 4)[:, :, :3]

        # Preprocess
        arr = arr / 255.0
        arr = np.resize(arr, (*input_shape, 3))
        return np.expand_dims(arr, axis=0)

    def create_results_chart(self, results):
        """Create a chart showing results from all models."""
        figure = Figure(figsize=(10, 6))
        canvas = FigureCanvas(figure)

        ax = figure.add_subplot(111)
        x = np.arange(len(results[0]['labels']))
        width = 0.8 / len(results)

        for i, result in enumerate(results):
            offset = (i - len(results) / 2 + 0.5) * width
            bars = ax.bar(x + offset, result['probs'], width,
                          label=result['model_name'])

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0.05:  # Only show labels for significant probabilities
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                            f'{height:.2f}', ha='center', va='bottom')

        ax.set_xticks(x)
        ax.set_xticklabels(results[0]['labels'], rotation=45, ha='right')
        ax.set_ylabel('Probability')
        ax.set_title('Model Predictions Comparison')
        ax.legend()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle='--', alpha=0.7)

        figure.tight_layout()
        return canvas

    def run_selected(self):
        """Run selected models on selected images."""
        # Clear previous results
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)

        # Get selected images
        selected_images = []
        for i in range(self.image_grid.count()):
            widget = self.image_grid.itemAt(i).widget()
            if isinstance(widget, ImageLabel) and widget.selected:
                selected_images.append(widget)

        if not selected_images:
            self.show_status("Please select at least one image")
            return

        # Process each selected image
        for image_widget in selected_images:
            results = []

            # Run each model
            for model_name in self.model_manager.get_model_names():
                model, labels = self.model_manager.get_model(model_name)
                if model is None:
                    continue

                try:
                    # Prepare image
                    image = self.prepare_image(image_widget._pixmap,
                                               model.input_shape[1:3])

                    # Run prediction
                    output = model.predict(image)
                    results.append({
                        'model_name': model_name,
                        'probs': output[0],
                        'labels': ([labels.get(str(i), f"Class {i}")
                                    for i in range(len(output[0]))]
                                   if labels else [f"Class {i}"
                                                   for i in range(len(output[0]))])
                    })

                except Exception as e:
                    self.show_status(f'Error running model {model_name}: {str(e)}')

            # Create and add results chart
            if results:
                chart = self.create_results_chart(results)
                self.results_layout.addWidget(chart)

        # Switch to results tab
        self.tab_widget.setCurrentWidget(self.results_tab)

    def show_status(self, message):
        """Show status message (could be enhanced with a status bar)."""
        print(message)  # For now, just print to console


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CNNViewer()
    window.show()
    sys.exit(app.exec_())