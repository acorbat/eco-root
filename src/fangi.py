import cv2
import numpy as np
from skimage.filters import frangi, hessian
import os

def apply_frangi_filter(video_path, sigmas=range(1, 4), black_ridges=False):
    """
    Apply Frangi filter to enhance vessel/root-like structures in a video.
    Processes the video in-place, overwriting the original file.
    
    Args:
        video_path: Path to the video file to process
        sigmas: Range of sigma values for vessel thickness detection (in pixels)
        black_ridges: Set to True for dark structures, False for bright structures
    """
    import tempfile
    import shutil
    
    print(f"Applying Frangi filter to: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Write to a temporary file first
    temp_fd, temp_path = tempfile.mkstemp(suffix='.AVI')
    os.close(temp_fd)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (w, h), isColor=False)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Normalize to float (0.0 to 1.0) for skimage
        img_float = gray.astype(np.float32) / 255.0
        
        # Apply Frangi filter to enhance tubular structures
        filt_frangi = frangi(img_float, sigmas=sigmas, black_ridges=black_ridges)
        
        # Normalize and scale to 0-255 range
        if np.max(filt_frangi) > 0:
            filt_frangi = filt_frangi * (255.0 / np.max(filt_frangi))
        
        final_frame = filt_frangi.astype(np.uint8)
        out.write(final_frame)
        frame_count += 1
        
    cap.release()
    out.release()
    
    # Replace original file with processed file
    shutil.move(temp_path, video_path)
    print(f"Frangi filter applied. Processed {frame_count} frames: {video_path}")


if __name__ == "__main__":
    apply_frangi_filter("data/videos/001.AVI")