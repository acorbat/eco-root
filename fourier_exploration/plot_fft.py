import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_fft_magnitude(image_path):
    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Compute 2D Fourier Transform and shift zero freq to center
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)

    # Magnitude spectrum (log scale for visibility)
    magnitude_spectrum = 20 * np.log(np.abs(Fshift) + 1)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Fourier Magnitude Spectrum (log scale)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_fft_magnitude.py <path_to_image>")
        sys.exit(1)
    plot_fft_magnitude(sys.argv[1])
