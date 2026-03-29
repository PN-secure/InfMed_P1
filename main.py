import matplotlib.pyplot as plt
import sys
from calculators import *
import numpy as np
from PIL import Image

def img2bitmap(image_path):
    image = Image.open(image_path).convert("L")
    image = np.array(image)
    return image / 255

plt.subplot(1,2,1)
img_bitmap = img2bitmap(sys.argv[1])
plt.imshow(img_bitmap, cmap="gray")
plt.subplot(1,2,2)
plt.imshow(radon_transform(img_bitmap, np.linspace(0,180,180)), cmap="gray", aspect="auto")
plt.show()
