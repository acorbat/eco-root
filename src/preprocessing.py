import SimpleITK as sitk
import numpy as np
from skimage.filters import frangi


def frangi_3d_filter(volume, sigmas_mm=(1.0, 2.0, 3.0), black_ridges=False, 
                     alpha=0.5, beta=0.5, gamma=15):
    """
    Apply 3D Frangi vesselness filter to enhance tubular root structures.
    Handles anisotropic voxel spacing by scaling sigmas per dimension.
    
    Args:
        volume: SimpleITK.Image
            Input 3D volume with spacing metadata. Must have proper spacing set.
        sigmas_mm: tuple of float
            Sigma values in physical units (millimeters) representing the range of 
            root/vessel diameters to detect. Default: (1.0, 2.0, 3.0) for 1-3mm roots.
        black_ridges: bool
            True for dark structures on bright background, False for bright structures.
            Default: False (for bright roots in ultrasound).
        alpha: float
            Plate-like structures threshold (lower = less sensitive to plates).
            Default: 0.5
        beta: float  
            Blob-like structures threshold (lower = less sensitive to blobs).
            Default: 0.5
        gamma: float
            Noise threshold for second-order structureness (higher = less noise sensitive).
            Default: 15 (good for noisy ultrasound)
    
    Returns:
        SimpleITK.Image
            Enhanced 3D volume with the same metadata as input.
            Bright values indicate high vesselness (tubular structure likelihood).
    """
    # Get physical spacing from the volume (x, y, z) in mm
    spacing = volume.GetSpacing()
    
    # Convert SimpleITK image to numpy array (z, y, x) ordering
    arr = sitk.GetArrayFromImage(volume)
    
    # Normalize to [0, 1] range for skimage processing
    arr_min = arr.min()
    arr_max = arr.max()
    arr_range = arr_max - arr_min
    
    if arr_range == 0:
        print("Warning: Volume has uniform intensity. Returning original volume without Frangi filtering.")
        return volume
    
    arr_norm = (arr - arr_min) / arr_range
    
    # Handle anisotropic spacing for Frangi filter
    # spacing is (x, y, z) but array is (z, y, x)
    # We need to work in pixel space accounting for different resolutions
    spacing_array = np.array([spacing[2], spacing[1], spacing[0]])  # reorder to (z, y, x)
    
    # Convert sigmas from mm to pixels for EACH dimension
    # For highly anisotropic data, we scale differently per axis
    sigmas_pixels_list = []
    for sigma_mm in sigmas_mm:
        # Scale sigma based on each dimension's spacing
        sigma_pixels = sigma_mm / spacing_array
        sigmas_pixels_list.append(sigma_pixels)
    
    print(f"Volume spacing: {spacing} mm/pixel (x,y,z)")
    print(f"Array shape: {arr_norm.shape} (z,y,x)")
    print(f"Frangi sigmas in mm: {sigmas_mm}")
    print(f"Frangi parameters: alpha={alpha}, beta={beta}, gamma={gamma}")
    
    # Apply Frangi filter at each scale and take maximum response
    enhanced = np.zeros_like(arr_norm)
    
    for i, sigma_pixels in enumerate(sigmas_pixels_list):
        print(f"  Scale {i+1}/{len(sigmas_mm)}: sigma={sigmas_mm[i]}mm = {sigma_pixels} pixels (z,y,x)")
        
        # Apply Frangi at this scale
        # Note: skimage frangi expects isotropic sigma, so we approximate with mean
        sigma_mean = np.mean(sigma_pixels)
        scale_response = frangi(
            arr_norm,
            sigmas=[sigma_mean],
            black_ridges=black_ridges,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            mode='reflect'
        )
        
        # Keep maximum response across scales
        enhanced = np.maximum(enhanced, scale_response)
    
    # Convert back to SimpleITK image and preserve all metadata
    result = sitk.GetImageFromArray(enhanced)
    result.CopyInformation(volume)
    
    return result

