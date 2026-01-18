import cv2
import os

def export_frames_every_n(video_path, output_dir, n=100):
    """
    Extracts every n-th frame from a video and saves it as an image.
    
    Args:
        video_path (str): Path to the .AVI video file.
        output_dir (str): Folder where extracted frames will be saved.
        n (int): Interval between saved frames (default: 3).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Save every nth frame
        if frame_idx % n == 0:
            filename = os.path.join(output_dir, f"frame_{saved_idx:04d}.png")
            cv2.imwrite(filename, frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"✅ Done! Saved {saved_idx} frames to '{output_dir}'.")

# Example usage
if __name__ == "__main__":
    video_path = "data/videos/003.avi"   # path to your .AVI file
    output_dir = "frames_output"     # folder to save frames
    export_frames_every_n(video_path, output_dir, n=7)
