import numpy as np

import numpy as np


def radon_transform(image, angles):
    height, width = image.shape

    # przekątna = maksymalny zasięg
    diag = int(np.sqrt(height ** 2 + width ** 2))

    # sinogram
    sinogram = np.zeros((diag, len(angles)))

    # środek obrazu
    cx, cy = width // 2, height // 2

    for a_idx, theta in enumerate(angles):
        theta_rad = np.deg2rad(theta)
        cos_t = np.cos(theta_rad)
        sin_t = np.sin(theta_rad)

        for y in range(height):
            for x in range(width):

                # przesunięcie do środka
                x_shift = x - cx
                y_shift = y - cy

                # obliczenie t
                t = int(x_shift * cos_t + y_shift * sin_t)

                t_idx = t + diag // 2

                if 0 <= t_idx < diag:
                    sinogram[t_idx, a_idx] += image[y, x]

    return sinogram