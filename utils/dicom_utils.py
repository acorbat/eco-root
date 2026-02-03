import pydicom
import numpy as np
import cv2
import os

def  extract_video(dicom_file_path):
    """
    Extracts video from a DICOM file and saves it as an AVI file.

    input: dicom_file_path: str
    output: video_path: str
    """

    # Check if file has .dcm extension
    if not dicom_file_path.lower().endswith('.dcm'):
        raise ValueError(f"File must have .dcm extension. Got: {dicom_file_path}")
    
    dicom_data = pydicom.dcmread(dicom_file_path)

    # Check if the DICOM file contains pixel data
    if 'PixelData' not in dicom_data:
        raise ValueError("DICOM file does not contain pixel data.")

    frames = dicom_data.pixel_array

    # Get frame dimensions
    # grayscale
    if len(frames.shape) == 3:
        num_frames, height, width = frames.shape
    #color
    if len(frames.shape) == 4:
        num_frames, height, width, _ = frames.shape
    else:
        raise ValueError("Unsupported frame shape in DICOM file.")

    # Create output directory if it doesn't exist
    output_dir = "data/videos"
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, os.path.splitext(os.path.basename(dicom_file_path))[0] + ".AVI")

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = float(dicom_data.get('CineRate', 30))  # Use 30 fps by default if not present
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height), isColor=False)

    # Write frames to video 
    for i in range(num_frames):
        frame = frames[i]
        if len(frame.shape) == 3:  # Convert color to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.write(frame.astype(np.uint8))
    out.release()

    return video_path

def rename_dicom_files_sequentially(dicom_folder):
    """
    Rename DICOM files in a folder sequentially based on their acquisition time.

    Args:
        dicom_folder (str): Path to the folder containing DICOM files.
    """

    print(f"Renaming files in {dicom_folder} based on acquisition time...")

    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith(".dcm")]

    # Extract acquisition time and associate with filename
    file_times = []
    for f in dicom_files:
        ds = pydicom.dcmread(os.path.join(dicom_folder, f), stop_before_pixels=True)
        time = ds.get("AcquisitionTime", "000000")
        file_times.append((f, time))

    # Sort by time
    file_times.sort(key=lambda x: x[1])

    # Rename or number sequentially
    for idx, (f, t) in enumerate(file_times, 1):
        old_path = os.path.join(dicom_folder, f)
        new_filename = f"{idx:03d}.dcm"  
        new_path = os.path.join(dicom_folder, new_filename)
        os.rename(old_path, new_path)
        print(f"{f} -> {new_filename}")
