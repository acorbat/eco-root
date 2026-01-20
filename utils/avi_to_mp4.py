"""
AVI to MP4 Converter

Converts AVI video files to MP4 format using OpenCV.

Usage:
    python src/avi_to_mp4.py                    # Convert all AVIs in data/videos
    python src/avi_to_mp4.py path/to/video.avi  # Convert specific file
"""

import os
import sys
import cv2


def convert_avi_to_mp4(input_path: str, output_path: str = None) -> str:
    """
    Convert an AVI video file to MP4 format.
    
    Args:
        input_path: Path to the input AVI file
        output_path: Optional path for output MP4 file. 
                     If not provided, uses same name with .mp4 extension.
    
    Returns:
        Path to the output MP4 file
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.mp4"
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  Input: {input_path}")
    print(f"  Properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Create output video writer with MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Copy frames
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_count += 1
        
        # Progress indicator
        if frame_count % 50 == 0:
            print(f"  Progress: {frame_count}/{total_frames} frames", end='\r')
    
    cap.release()
    out.release()
    
    print(f"  Output: {output_path} ({frame_count} frames written)")
    return output_path


def convert_directory(input_dir: str, output_dir: str = None):
    """
    Convert all AVI files in a directory to MP4.
    
    Args:
        input_dir: Directory containing AVI files
        output_dir: Optional output directory. If not provided, saves to same location.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    # Find all AVI files
    avi_files = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith('.avi')]
    
    if not avi_files:
        print(f"No AVI files found in {input_dir}")
        return
    
    print(f"Found {len(avi_files)} AVI file(s) to convert")
    print("=" * 50)
    
    for i, filename in enumerate(avi_files, 1):
        print(f"\n[{i}/{len(avi_files)}] Converting: {filename}")
        
        input_path = os.path.join(input_dir, filename)
        
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.mp4")
        else:
            output_path = None
        
        try:
            convert_avi_to_mp4(input_path, output_path)
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 50)
    print("Conversion complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Convert specific file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_avi_to_mp4(input_file, output_file)
    else:
        # Convert all AVIs in data/videos
        convert_directory("data/videos")
