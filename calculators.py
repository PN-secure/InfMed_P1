import numpy as np
import os
from PIL import Image
import pydicom

def bresenham(x0, y0, x1, y1):
    points = []

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    while True:
        points.append((x0, y0))

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x0 += sx

        if e2 < dx:
            err += dx
            y0 += sy

    return points

def radon_transform(image, angles, n_detectors, detector_span):
    height, width = image.shape
    n = n_detectors

    t_max = detector_span / 2

    t_vals = np.linspace(-t_max, t_max, n)

    sinogram = np.zeros((n, len(angles)))

    cx, cy = width // 2, height // 2

    for a_idx, theta in enumerate(angles):
        theta_rad = np.deg2rad(theta)

        # kierunek prostopadły (do przesuwania linii)
        dx = np.cos(theta_rad) 
        dy = np.sin(theta_rad)

        # wektor prostopadły do linii
        nx = -dy
        ny = dx
        L = max(width, height) * 2
        for t_idx, t in enumerate(t_vals):

            # punkt na linii
            x0 = int(cx + t * nx)
            y0 = int(cy + t * ny)

            # końce długiej linii (żeby przeciąć cały obraz)
            x1 = int(x0 + L * dx)
            y1 = int(y0 + L * dy)
            x2 = int(x0 - L * dx)
            y2 = int(y0 - L * dy)

            # piksele na linii
            line_points = bresenham(x1, y1, x2, y2)

            sum_val = 0

            for x, y in line_points:
                if 0 <= x < width and 0 <= y < height:
                    sum_val += image[y, x]

            sinogram[t_idx, a_idx] = sum_val

    return sinogram

# funkcja kernela do filtru - można uprościć
def kernel(size):
    h = np.zeros(size)
    center = size // 2

    for i in range(size):
        k = i - center

        if k == 0:
            h[i] = 1
        elif k % 2 == 0:
            h[i] = 0
        else:
            h[i] = -4 / (np.pi**2 * k**2)
    
    h = h / np.sum(np.abs(h))
    return h
        
def filter_s(sinogram):
    h = kernel(21)
    sinogram /= np.max(sinogram)
    filtered = np.zeros_like(sinogram)

    for i in range(sinogram.shape[1]):
        filtered[:, i] = np.convolve(sinogram[:, i], h, mode="same")

    return filtered

def backprojection(sinogram, angles, output_size):
    height, width = output_size
    diag = sinogram.shape[0]

    recon = np.zeros((height, width))
    cx, cy = width // 2, height // 2

    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    X = X - cx
    Y = Y - cy

    for a_idx, theta in enumerate(angles):
        t = (X * np.sin(np.deg2rad(theta)) +
             Y * np.cos(np.deg2rad(theta)))

        t_idx = t + diag // 2

        t0 = np.floor(t_idx).astype(int)
        t1 = t0 + 1

        valid = (t0 >= 0) & (t1 < diag)

        alpha = t_idx - t0

        recon[valid] += (
            (1 - alpha[valid]) * sinogram[t0[valid], a_idx] +
            alpha[valid] * sinogram[t1[valid], a_idx]
        )

    return recon / len(angles)


def load_image(path):
    ext = os.path.splitext(path)[1].lower()

    # --- DICOM ---
    if ext == ".dcm":
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)

        # normalizacja
        img -= img.min()
        img /= (img.max() + 1e-8)
        return img

    # --- JPG / PNG ---
    else:
        img = Image.open(path).convert("L")
        img = np.array(img).astype(np.float32) / 255.0
        return img