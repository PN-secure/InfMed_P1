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
        t = (X * np.cos(np.deg2rad(theta)) +
             Y * np.sin(np.deg2rad(theta)))

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
