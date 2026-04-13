from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import numpy as np
import sys
from PIL import Image

from calculators import *

class Worker(QThread):
    finished = pyqtSignal(object, object, object, object)

    def run(self):
        img = Image.open(sys.argv[1]).convert("L")
        img = np.array(img) / 255

        angles = np.linspace(0, 180, 180, endpoint=False)

        x = radon_transform(img, angles)
        fil = filter_s(x)
        rec = backprojection(fil, angles, img.shape)

        self.finished.emit(img, x, fil, rec)

class App(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tomograf komputerowy")
        self.resize(1400, 800)
        
        self.create_buttons()

        self.fig = Figure(figsize=(10, 3))
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.btn = QPushButton("Rysuj")
        self.btn.clicked.connect(self.start)
        self.btn.setMaximumSize(150,50)

        layout = QVBoxLayout()
        layout.addLayout(self.radio_layout) # dodanie przyciskow trybiu
        layout.addWidget(self.canvas) #dodanie wykresu
        layout.addWidget(self.btn) #dodanie przycisku rysowania
        layout.insertWidget(0,self.btn)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = None

    def start(self):
        self.btn.setEnabled(False)

        self.worker = Worker()
        self.worker.finished.connect(self.update_plot)
        self.worker.start()

    def update_plot(self, img, x, fil, rec):
        self.fig.clear()

        ax1 = self.fig.add_subplot(1, 4, 1)
        ax1.imshow(img, cmap="gray")
        ax1.set_title("Wejście")

        ax2 = self.fig.add_subplot(1, 4, 2)
        ax2.imshow(x, cmap="gray", aspect="auto")
        ax2.set_title("Transformata Radona")

        ax3 = self.fig.add_subplot(1, 4, 3)
        ax3.imshow(fil, cmap="gray", aspect="auto")
        ax3.set_title("Filtr")

        ax4 = self.fig.add_subplot(1, 4, 4)
        ax4.imshow(rec, cmap="gray")
        ax4.set_title("Backprojection")

        self.canvas.draw()
        self.btn.setEnabled(True)
    
    def create_buttons(self): # metoda tworzy grupę radiobuttonów
        self.iter_b = QRadioButton("Pokaż kroki")
        self.n_iter_b = QRadioButton("Nie pokazuj kroków")
        
        self.group = QButtonGroup()
        self.group.addButton(self.iter_b)
        self.group.addButton(self.n_iter_b)
        
        self.n_iter_b.setChecked(True)
        
        self.radio_layout = QVBoxLayout()
        self.radio_layout.addWidget(self.iter_b)
        self.radio_layout.addWidget(self.n_iter_b)


app = QApplication([])
window = App()
window.show()
app.exec()
