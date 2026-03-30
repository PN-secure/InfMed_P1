import matplotlib.pyplot as plt
import sys
from calculators import *
import numpy as np
from PIL import Image

def img2bitmap(image_path):
    image = Image.open(image_path).convert("L")
    image = np.array(image)
    return image / 255

def main():
    plt.subplot(1,3,1)
    img_bitmap = img2bitmap(sys.argv[1])
    plt.imshow(img_bitmap, cmap="gray")
    plt.subplot(1,3,2)
    x = radon_transform(img_bitmap, np.linspace(0,180,180))
    plt.imshow(x, cmap="gray", aspect="auto") # sinogram nieprzefiltrowany
    plt.subplot(1,3,3)
    plt.imshow(filter(x), cmap="gray", aspect="auto") #sinogram przefiltrowany
    plt.show()

if __name__ == "__main__":
    main()
