from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QRadioButton, QButtonGroup, QSlider, QLabel
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QCoreApplication, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import sys
import pydicom
from PIL import Image

from calculators import *

class Worker(QThread):
    finished = pyqtSignal(object, object, object, object, object, object)

    def __init__(self, delta_alpha, n_detectors, detector_span):
        super().__init__()
        self.delta_alpha = delta_alpha
        self.n_detectors = n_detectors
        self.detector_span = detector_span

    def run(self):
        path = sys.argv[1]
        img = load_image(path)

        angles = np.arange(0, 180, self.delta_alpha)

        sinogram = radon_transform(
            img,
            angles,
            self.n_detectors,
            self.detector_span
        )
        
        filtered = filter_s(sinogram)

        reconstructions = []

        for i in range(1, len(angles)+1):
            rec = backprojection(
                filtered[:, :i],
                angles[:i],
                img.shape
            )
            reconstructions.append(rec)

        self.finished.emit(img, sinogram, filtered, angles, img.shape, reconstructions)

class App(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tomograf komputerowy")
        self.resize(1400, 850)
        
        self.create_draw_button()
        self.create_radio_buttons()
        self.create_slider()
        self.create_controls()

        self.fig = Figure(figsize=(10, 3))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax1 = self.fig.add_subplot(1,4,1)
        self.ax2 = self.fig.add_subplot(1,4,2)
        self.ax3 = self.fig.add_subplot(1,4,3)
        self.ax4 = self.fig.add_subplot(1,4,4)

        layout = QVBoxLayout()
        layout.addLayout(self.radio_layout) # dodanie przyciskow trybiu
        layout.addWidget(self.slider)
        layout.addWidget(self.canvas) #dodanie wykresu
        layout.addLayout(self.controls_layout)
        layout.addWidget(self.btn) #dodanie przycisku rysowania
        layout.insertWidget(0,self.btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = None

    def start(self):
        self.btn.setEnabled(False)

        delta_alpha = self.delta_slider.value()
        n_detectors = self.n_slider.value()
        detector_span = self.l_slider.value()

        self.worker = Worker(delta_alpha, n_detectors, detector_span)
        self.worker.finished.connect(self.update_plot)
        self.worker.start()
    
    def update_iter_plot(self, value):
        if not hasattr(self, "reconstructions"):
            return

        rec = self.reconstructions[value - 1]

        self.draw_all(
            self.img,
            self.sinogram[:, :value],
            self.filtered[:, :value],
            rec
        )
    def draw_all(self, img, x, fil, rec):
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()

        self.ax1.imshow(img, cmap="gray")
        self.ax1.set_title("Wejście")

        self.ax2.imshow(x, cmap="gray", aspect="auto")
        self.ax2.set_title("Transformata Radona")

        self.ax3.imshow(fil, cmap="gray", aspect="auto")
        self.ax3.set_title("Filtr")

        self.ax4.imshow(rec, cmap="gray")
        self.ax4.set_title("Backprojection")

        self.canvas.draw()

        self.btn.setEnabled(True)
    
    def create_radio_buttons(self): # metoda tworzy grupę radiobuttonów
        self.iter_b = QRadioButton("Pokaż kroki")
        self.n_iter_b = QRadioButton("Nie pokazuj kroków")
        
        self.group = QButtonGroup()
        self.group.addButton(self.iter_b)
        self.group.addButton(self.n_iter_b)
        
        self.n_iter_b.setChecked(True)
        self.iter_b.toggled.connect(self.toggle_slider)
        self.radio_layout = QVBoxLayout()
        self.radio_layout.addWidget(self.iter_b)
        self.radio_layout.addWidget(self.n_iter_b)
        

    def create_draw_button(self): # utworzenie przycisku rysuj
        self.btn = QPushButton("Rysuj")
        self.btn.clicked.connect(self.start)
        self.btn.setMaximumSize(150,50)

    def create_slider(self): # utworzenie suwaka
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.blockSignals(True)
        self.slider.setValue(1)
        self.slider.blockSignals(False)
        self.slider.valueChanged.connect(self.update_iter_plot)
        self.slider.hide()
        self.slider.setMaximumSize(150,50)

    def create_controls(self):
        layout = QVBoxLayout()

        # delta alfa
        self.delta_label = QLabel("Krok delta alfa: 1")
        self.delta_slider = QSlider(Qt.Orientation.Horizontal)
        self.delta_slider.setMinimum(1)
        self.delta_slider.setMaximum(10)
        self.delta_slider.setValue(1)
        self.delta_slider.valueChanged.connect(
            lambda v: self.delta_label.setText(f"Krok delta alfa: {v}")
        )

        # n
        self.n_label = QLabel("Liczba detektorów (n): 180")
        self.n_slider = QSlider(Qt.Orientation.Horizontal)
        self.n_slider.setMinimum(10)
        self.n_slider.setMaximum(300)
        self.n_slider.setValue(180)
        self.n_slider.valueChanged.connect(
            lambda v: self.n_label.setText(f"Liczba detektorów (n): {v}")
        )

        # l
        self.l_label = QLabel("Rozpiętość detektorów (l): 200")
        self.l_slider = QSlider(Qt.Orientation.Horizontal)
        self.l_slider.setMinimum(50)
        self.l_slider.setMaximum(500)
        self.l_slider.setValue(200)
        self.l_slider.valueChanged.connect(
            lambda v: self.l_label.setText(f"Rozpiętość detektorów (l): {v}")
        )

        layout.addWidget(self.delta_label)
        layout.addWidget(self.delta_slider)

        layout.addWidget(self.n_label)
        layout.addWidget(self.n_slider)

        layout.addWidget(self.l_label)
        layout.addWidget(self.l_slider)

        self.controls_layout = layout

    def toggle_slider(self, checked): # ukrycie lub wyświetlenie suwaka
        if checked:
            self.slider.show()
        else:
            self.slider.hide()

    def update_plot(self, img, sinogram, filtered, angles, shape, reconstructions):
        self.img = img
        self.sinogram = sinogram
        self.filtered = filtered
        self.angles = angles
        self.shape = shape
        self.reconstructions = reconstructions

        self.slider.setMaximum(len(angles))

        if self.iter_b.isChecked():
            self.slider.setEnabled(True)
            self.update_iter_plot(self.slider.value())
        else:
            self.slider.setEnabled(False)
            self.draw_all(img, sinogram, filtered, reconstructions[-1])

        self.btn.setEnabled(True)

app = QApplication([])
window = App()
window.show()
app.exec()