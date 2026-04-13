from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QRadioButton, QButtonGroup, QSlider
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QCoreApplication, QTimer
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import sys
from PIL import Image

from calculators import *

class Worker(QThread):
    finished = pyqtSignal(object, object, object, object, object)

    def run(self):
        img = Image.open(sys.argv[1]).convert("L")
        img = np.array(img) / 255

        angles = np.linspace(0, 180, 180, endpoint=False)

        x = radon_transform(img, angles)
        fil = filter_s(x)
        rec = backprojection(fil, angles, img.shape)

        self.finished.emit(img, x, fil, angles, img.shape)

class App(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tomograf komputerowy")
        self.resize(1400, 850)
        
        self.create_draw_button()
        self.create_radio_buttons()
        self.create_slider()

        self.fig = Figure(figsize=(10, 3))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax1 = self.fig.add_subplot(1,4,1)
        self.ax2 = self.fig.add_subplot(1,4,2)
        self.ax3 = self.fig.add_subplot(1,4,3)
        self.ax4 = self.fig.add_subplot(1,4,4)

        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._update_delayed)

        layout = QVBoxLayout()
        layout.addLayout(self.radio_layout) # dodanie przyciskow trybiu
        layout.addWidget(self.slider)
        layout.addWidget(self.canvas) #dodanie wykresu
        layout.addWidget(self.btn) #dodanie przycisku rysowania
        layout.insertWidget(0,self.btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = None

    def start(self):
        self.btn.setEnabled(False)
        #if self.iter_b.isChecked():
        #    self.btn.setEnabled(False)
        #else:
        self.worker = Worker()
        self.worker.finished.connect(self.update_plot)
        self.worker.start()
    
    def update_iter_plot(self, value):
        if value == 0:
            return
        
        partial_angles = self.angles[:value]
        partial_sinogram = self.filtered[:, :value]
        
        self._pending_angles = partial_angles
        self._pending_sinogram = partial_sinogram

        self.timer.start(50)
        
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

        self.canvas.draw_idle()

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
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.blockSignals(True)
        self.slider.setValue(1)
        self.slider.blockSignals(False)
        self.slider.valueChanged.connect(self.schedule_update)
        self.slider.hide()
        self.slider.setMaximumSize(150,50)

    def schedule_update(self, value):
        self._pending_value = value
        self.timer.start(80)

    def _update_delayed(self):
        if not hasattr(self, "_pending_angles"):
            return

        rec = backprojection(
            self._pending_sinogram,
            self._pending_angles,
            self.shape
        )

        self.draw_all(
            self.img,
            self.sinogram[:, :len(self._pending_angles)],
            self._pending_sinogram,
            rec
        )

    def toggle_slider(self, checked): # ukrycie lub wyświetlenie suwaka
        if checked:
            self.slider.show()
        else:
            self.slider.hide()

    def update_plot(self, img, sinogram, filtered, angles, shape):
        self.img = img
        self.sinogram = sinogram
        self.filtered = filtered
        self.angles = angles
        self.shape = shape
        
        self.slider.setMaximum(len(angles))
        self.slider.setEnabled(True)

        if self.iter_b.isChecked():
            self.slider.setEnabled(True)
            self.update_iter_plot(self.slider.value())
        else:
            self.slider.setEnabled(False)
            rec = backprojection(filtered, angles, shape)
            self.draw_all(img, sinogram, filtered, rec)

        self.btn.setEnabled(True)

app = QApplication([])
window = App()
window.show()
app.exec()
