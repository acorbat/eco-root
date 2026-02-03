import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import SimpleITK as sitk
from utils.dicom_utils import extract_video, rename_dicom_files_sequentially
from volumeReconstructor import VolumeReconstructor
from volumeRegistrator import VolumeRegistrator
from preprocessing import frangi_3d_filter

# # ------------------- SET UP ------------------
# # Extract videos from DICOM files

# print("\n" + "="*50)
# print("VIDEO EXTRACTION")
# print("="*50)


# input_dir = 'data/ultrasound/raices/1'

# rename_dicom_files_sequentially(input_dir)

# for filename in os.listdir(input_dir):
        
#     input_path = os.path.join(input_dir, filename)
#     if os.path.isfile(input_path):
#         try:
#             video_path = extract_video(input_path)
#             print(f"Video extracted in: {video_path}")
#         except Exception as e:
#             print(f"Could not extract video from {input_path}: {e}")

extracted_videos_dir = 'data/videos'

# -------------------- VOLUME RECONSTRUCTION ------------------
# Generate volume reconstruction for each view
# stacks each frame into a 3D volume where each cut is a z plane. 
# It also crops each frame so only the ultrasound region is kept. 

print("\n" + "="*50)
print("VOLUME RECONSTRUCTION")
print("="*50)


volumes_output_dir = 'data/volumes'
if not os.path.exists(volumes_output_dir):
    os.makedirs(volumes_output_dir)

mask_path = "data/crop_masks/mascara_no_borders.png" 

for filename in os.listdir(extracted_videos_dir):
    video_path = os.path.join(extracted_videos_dir, filename)
    if os.path.isfile(video_path) and filename.endswith('.AVI'):
        try:
            print(f"\nProcessing video: {filename}")
    
            # Create reconstructor and extract volume
            # Using real physical spacing in mm: (x, y, z)
            reconstructor = VolumeReconstructor(video_path, mask_path=mask_path, sampling_rate=6, voxel_spacing=(0.187, 0.188, 4.167))
            volume = reconstructor.create_volume()
    
            # Generate output filename
            base_name = os.path.splitext(filename)[0]  # Remove .AVI extension
            output_filename = f"volume_{base_name}.nii.gz"
            output_path = os.path.join(volumes_output_dir, output_filename)
    
            # Save volume
            sitk.WriteImage(volume, output_path)
            print(f"Volume saved to: {output_path}")
    
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# -------------------- VOLUME PREPROCESSING ------------------
print("\n" + "="*50)
print("PREPROCESSING")
print("="*50)

for filename in os.listdir(volumes_output_dir):
    if not filename.endswith('.nii.gz'):
        continue
    img = sitk.ReadImage(os.path.join(volumes_output_dir, filename))
    print(f"\nProcessing {filename} with Frangi filter...")
    
    # Apply Frangi with optimized parameters for ultrasound root detection
    img_enhanced = frangi_3d_filter(
        img, 
        sigmas_mm=(0.2, 0.3, 0.4, 0.5, 0.6),  # detect roots 1-3mm diameter
        black_ridges=False,          # bright roots
        alpha=1,                   # LOW = strict rejection of plate-like structures
        beta=1,                    # LOW = strict rejection of blob-like structures  
        gamma=15                     # noise threshold (higher = less sensitive to noise)
    )
    
    sitk.WriteImage(img_enhanced, os.path.join(volumes_output_dir, filename))
    print(f"Frangi-enhanced volume saved to: {os.path.join(volumes_output_dir, filename)}")    

# -------------------- REGISTRATION ------------------
# Perform volume registration and fusion

print("\n" + "="*50)
print("VOLUME REGISTRATION AND FUSION")
print("="*50)

volumes_dir = "data/volumes"  
registrator = VolumeRegistrator(volumes_dir)
fused_img = registrator.process_volumes()

# -------------------- SEGMENTATION ------------------
print("\n" + "="*50)
print("VOLUME SEGMENTATION")
print("="*50)

print("Applying segmentation to the final fused volume...")

# Use the fused volume for segmentation
volume = sitk.ReadImage(fused_img)

# Gaussian filter (optional - uncomment if needed for noise reduction)
# volume = sitk.GradientMagnitudeRecursiveGaussian(volume, sigma=0.1)

# Median filter - removes noise
print("Applying median filter...")
median = sitk.MedianImageFilter()
filtered_volume = median.Execute(volume)

# Otsu thresholding - segments the object from the background
print("Applying Otsu thresholding...")
otsu_filter = sitk.OtsuThresholdImageFilter()
otsu_filter.SetInsideValue(0)  # sets the pixel value for the background as 0
otsu_filter.SetOutsideValue(1)  # sets the pixel value for the object as 1
seg = otsu_filter.Execute(filtered_volume)

# Optional morphological operations (uncomment if needed)
seg = sitk.BinaryFillhole(seg)
seg = sitk.BinaryMorphologicalOpening(seg)

# Write output files
print("Saving segmented volumes...")
outVol = "data/results/final_volume.nii.gz"
outSeg = "data/results/final_volume_segmentation.nii.gz"

sitk.WriteImage(filtered_volume, outVol)
sitk.WriteImage(seg, outSeg)

print(f"Filtered final volume saved to: {outVol}")
print(f"Segmented final volume saved to: {outSeg}")

            



