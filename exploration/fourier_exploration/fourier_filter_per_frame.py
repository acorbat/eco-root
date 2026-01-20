"""Small test script: load a NIfTI volume, extract the axial (Z) middle slice and save it.

This file replaces the previous one-line placeholder with a tiny utility that uses
SimpleITK to read the volume, converts it to a NumPy array (shape: z,y,x), picks
the middle axial slice, normalizes it to 0-255 and writes a PNG to the results/ folder.
"""

import os
import SimpleITK as sitk
import numpy as np
import cv2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Path to the input volume (default from the original file)
volume_path = 'data/volumes/volume_003.nii.gz'


def save_mid_axial_slice(volume_path: str, out_path: str | None = None, normalize: bool = True) -> str:
	"""Read a volume and save the axial middle slice as a PNG.

	- volume_path: path to a .nii or .nii.gz file
	- out_path: optional output PNG path. If None, writes to results/mid_axial_<basename>.png
	- normalize: scale slice intensities to 0-255 based on min/max

	Returns the path written.
	"""
	if not os.path.exists(volume_path):
		raise FileNotFoundError(f"Volume not found: {volume_path}")

	img = sitk.ReadImage(volume_path)
	arr = sitk.GetArrayFromImage(img)  # shape: (z, y, x)

	# If the image is 4D (e.g., multiple timepoints or channels), try to use the first volume
	if arr.ndim == 4:
		arr = arr[0]

	z_mid = arr.shape[0] // 2
	slice_axial = arr[z_mid, :, :]

	# Normalize to 0-255 uint8 for saving as PNG
	if normalize:
		mn = np.nanmin(slice_axial)
		mx = np.nanmax(slice_axial)
		if mx > mn:
			slice_norm = (slice_axial - mn) / (mx - mn)
		else:
			slice_norm = np.zeros_like(slice_axial, dtype=float)
		slice_uint8 = (slice_norm * 255.0).astype(np.uint8)
	else:
		# Clip to uint8 range if necessary
		slice_uint8 = np.clip(slice_axial, 0, 255).astype(np.uint8)

	if out_path is None:
		base = os.path.splitext(os.path.basename(volume_path))[0]
		out_dir = 'test'
		os.makedirs(out_dir, exist_ok=True)
		out_path = os.path.join(out_dir, f'mid_axial_{base}.png')
	else:
		os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

	# OpenCV expects (h, w) for single-channel images and writes as BGR for color.
	success = cv2.imwrite(out_path, slice_uint8)
	if not success:
		raise IOError(f"Failed to write image to {out_path}")

	print(f"Saved axial middle slice (index={z_mid}) to: {out_path}")
	return out_path

def remove_horizontal_reverb_fft_smooth(
    img,
    prominence=0.6,
    max_peaks=3,
    notch_sigma=2.5,
    notch_strength=0.95,
    min_freq_sep=4,
    debug=True
):
    """
    Smooth Gaussian notch filtering for horizontal reverberation (works on a single B-mode frame).
    - img: 2D float image normalized to [0..1]
    - prominence: relative prominence threshold for peaks in the vertical FFT profile (0..1)
    - max_peaks: maximum number of reverb frequencies to remove
    - notch_sigma: Gaussian sigma (in frequency rows). Accepts floats.
    - notch_strength: fraction to suppress at notch center (0..1). 0.95 -> 95% suppression.
    - min_freq_sep: minimum separation (in rows) between peaks to consider distinct.
    """
    H, W = img.shape
    # First you compute the fft, then you shift the zero frequency component to the center, so the low frequencies are in the center and the high frequencies in the outskirts
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(F)) ## calculates the log-magnitude spectrum, forgetting about the phase information

    # vertical profile: mean across columns -> strong vertical peaks correspond to horizontal stripes
    vert_profile = mag.mean(axis=1)
    center = H // 2

    # detect peaks in the vertical profile
    # convert prominence to absolute scale
    prom_abs = prominence * (vert_profile.max() - vert_profile.min())
    peaks, props = find_peaks(vert_profile, prominence=prom_abs, distance=min_freq_sep)
    # sort peaks by prominence and keep top ones
    if peaks.size > 0:
        prominences = props["prominences"]
        order = np.argsort(prominences)[::-1]
        peaks = peaks[order][:max_peaks]
    else:
        peaks = np.array([], dtype=int)

    # ignore peaks too close to DC
    peaks = np.array([p for p in peaks if abs(p - center) > 2])
    # ensure integer dtype for indexing (find_peaks sometimes yields float-ish types
    # after list comprehensions or ordering operations)
    if peaks.size > 0:
        peaks = np.asarray(peaks, dtype=int)
    else:
        peaks = np.array([], dtype=int)

    # build smooth notch mask (float)
    u = np.arange(H) - center
    v = np.arange(W) - (W // 2)
    U, V = np.meshgrid(u, v, indexing='ij')  # shape H x W

    mask = np.ones_like(F, dtype=np.float32)
    for p in peaks:
        freq_row = p - center  # signed offset from DC
        # gaussian in vertical freq dimension only (same for all columns)
        g = np.exp(-0.5 * ((U - freq_row) ** 2) / (notch_sigma ** 2))
        # convert gaussian to a notch multiplier: 1 - strength * g
        mask *= (1.0 - notch_strength * g).astype(np.float32)

    # apply mask smoothly
    F_filtered = F * mask

    # inverse FFT
    img_rec = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    img_rec = np.abs(img_rec)

    # normalize
    img_rec -= img_rec.min()
    if img_rec.max() > 0:
        img_rec /= img_rec.max()

    # optional post-processing to reduce remaining banding
    img_med = cv2.medianBlur((img_rec * 255).astype(np.uint8), ksize=3)
    img_med = img_med.astype(np.float32) / 255.0

    if debug:
        plt.figure(figsize=(14,5))
        plt.subplot(1,4,1)
        plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')

        plt.subplot(1,4,2)
        plt.imshow(mag, cmap='magma'); plt.title("FFT magnitude (log)"); plt.axis('off')

        plt.subplot(1,4,3)
        # show vertical profile and detected peaks
        plt.plot(vert_profile[::-1])  # reversed so top-of-image -> left in plot
        plt.scatter([(H-1 - p) for p in peaks], vert_profile[peaks], color='red')
        plt.title("Vertical FFT profile (peaks marked)")

        plt.subplot(1,4,4)
        plt.imshow(img_med, cmap='gray'); plt.title("Filtered + median"); plt.axis('off')

        plt.tight_layout()
        plt.show()

    return img_med, peaks


def dereverb_1d_cepstrum(img, qmin=8, qmax=60, alpha=0.7):
    """
    Simple 1D cepstral dereverberation along columns (beam direction).
    Works best after initial FFT filtering.
    
    - qmin/qmax: quefrency window for echo peak search (in samples)
    - alpha: strength of echo suppression (0..1)
    """
    H, W = img.shape
    out = np.zeros_like(img)
    # use NumPy's Hanning window (scipy.signal.hann may not be available in some SciPy builds)
    window = np.hanning(H)

    for x in range(W):
        a_line = img[:, x] * window
        spec = np.fft.fft(a_line)
        log_mag = np.log1p(np.abs(spec))
        cep = np.fft.ifft(log_mag).real

        # Find echo peak in cepstrum within [qmin, qmax]
        segment = cep[qmin:qmax]
        q_peak = np.argmax(segment) + qmin

        # Build comb filter in frequency domain to cancel echoes
        comb = 1 - alpha * np.exp(-1j * 2 * np.pi * q_peak * np.arange(H) / H)
        spec_filt = spec * comb

        # Reconstruct filtered A-line
        a_filt = np.fft.ifft(spec_filt).real
        out[:, x] = np.clip(a_filt, 0, None)

    out -= out.min()
    if out.max() > 0:
        out /= out.max()
    return out





if __name__ == '__main__':
	# ---------------------------
# Usage: change `path` as needed
# ---------------------------
    path = "test/mid_frame_003.png"  # <-- your image filename
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found at path: " + path)

    imgf = img.astype(np.float32)
    imgf /= imgf.max()

    # call the improved filter
    cleaned, detected_peaks = remove_horizontal_reverb_fft_smooth(
        imgf,
        prominence=0.04,      # raise to be more selective (0..1)
        max_peaks=9,
        notch_sigma=9.0,      # float allowed and recommended
        notch_strength=1,  # how strongly to suppress the peak (0..1)
        debug=True
    )

    # save and show comparison
    cv2.imwrite("ultrasound_cleaned_smooth.png", (cleaned * 255).astype(np.uint8))
    print("Saved ultrasound_cleaned_smooth.png, detected peak rows (indices):", detected_peaks)
    plt.figure(figsize=(8,6))
    plt.subplot(1,2,1); plt.imshow(imgf, cmap='gray'); plt.title("Original"); plt.axis('off')
    plt.subplot(1,2,2); plt.imshow(cleaned, cmap='gray'); plt.title("Smoothed Notch Result"); plt.axis('off')
    plt.show()

    # # Apply after your smooth notch result:
    # dereverb = dereverb_1d_cepstrum(cleaned, qmin=5, qmax=50, alpha=1)

    # cv2.imwrite("notch_filter_frame_003.png", (dereverb*255).astype(np.uint8))

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.imshow(cleaned, cmap='gray'); plt.title("After FFT notch")
    # plt.axis('off')
    # plt.subplot(1,2,2)
    # plt.imshow(dereverb, cmap='gray'); plt.title("After notch + cepstral dereverb")
    # plt.axis('off')
    # plt.tight_layout()
    # plt.show()
