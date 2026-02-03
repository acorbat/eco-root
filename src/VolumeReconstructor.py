import numpy as np
import cv2
import SimpleITK as sitk
from skimage import morphology
import scipy.ndimage as nd
import os.path

class VolumeReconstructor:
    def __init__(self, video_path, mask_path, sampling_rate=3, voxel_spacing=(1,1,5)):
        """
        Initialize VolumeReconstructor with basic parameters
        
        Args:
            video_path: str
            Path to the input video file
            mask_path: str
            Path to the mask image file
            sampling_rate: int
            Number of samples per second (default: 3)
        """
        # Store input paths and parameters
        self.video_path = video_path
        self.mask_path = mask_path
        self.sampling_rate = sampling_rate
        self.voxel_spacing = voxel_spacing
        
        # Load the mask
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Processing parameters (with defaults from original code)
        self.start_frame = 0
        self.end_frame = -1
        self.equalize = False
        
        # Results will be stored here
        self.volume = None
        self.segmented_volume = None
    
    # set up methods

    def extract_video_parameters(self):
        """
        Extract video parameters like FPS and total number of frames
        
        Returns:
            fps: int
            total_frames: int
        """
        cap = cv2.VideoCapture(self.video_path)
        
        # Extract video parameters (same as original code)
        fps = cap.get(cv2.CAP_PROP_FPS)  # gets video's fps rate
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # gets total number of frames in the video

        # Close the video capture
        cap.release()
        
        # Print info (same as original)
        print(f"FPS: {fps}")
        print(f"Total number of frames: {total_frames}")
        
        return fps, total_frames

    def get_frames_to_sample(self):
        """
        Calculate which frames to sample based on the sampling rate
        
        Returns:
            list: list[int]
            List of frame indices to sample
        """
        fps, total_frames = self.extract_video_parameters()
    
        if self.end_frame == -1:
            self.end_frame = int(total_frames - 1)
        
        step_sampling = int(fps // self.sampling_rate)
        frames_to_sample = list(range(self.start_frame, self.end_frame + 1, step_sampling))
        
        return frames_to_sample

    def extract_mask_bounding_box(self):
        """
        Extract mask bounding box - finds the min and max coordinates where the mask is white
        
        Returns:
            tuple: (min_coords, max_coords, size)
                min_coords: numpy array with minimum coordinates [y, x]
                max_coords: numpy array with maximum coordinates [y, x] 
                size: numpy array with size [height, width]
        """
        # Find all white pixels in the mask (same as original code)
        white_pixels = np.argwhere(self.mask == 255)
        
        if len(white_pixels) == 0:
            raise ValueError("Mask appears to be empty - no white pixels found")
        
        # Find min and max coordinates where the mask is white
        min_coords = np.min(white_pixels, axis=0)
        max_coords = np.max(white_pixels, axis=0)
        
        # Calculate size
        size = max_coords - min_coords
        
        # Print info (same as original)
        print(f"Mask size: {size}")
        
        return min_coords, max_coords, size

    # reconstruction methods

    def create_volume(self):
        """
        Create volumetric container and extract/crop frames from video
        
        Returns:
            numpy.ndarray: 3D volume with cropped frames (z, y, x)
        """
        # Get the frames to sample and mask bounding box
        frames_to_sample = self.get_frames_to_sample()
        min_coords, max_coords, size = self.extract_mask_bounding_box()
        fps, total_frames = self.extract_video_parameters()
        
        # Create volumetric image container
        # array to store the sampled frames cropped according to the mask size
        vol = np.zeros(shape=(len(frames_to_sample), size[0], size[1]), dtype=float)
        
        print(f"Created volumetric container with shape: {vol.shape}")
        
        cap = cv2.VideoCapture(self.video_path)
        
        i = 0  # current frame index
        z = 0  # volume depth index
        ret = True
        
        while i < total_frames and ret:
            ret, frame = cap.read()
            
            if not ret:
                break
                            
            if len(frame.shape) == 3:
                # Color image - convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                # Already grayscale
                gray = frame
            
            if i in frames_to_sample:
                if self.equalize:
                    gray = cv2.equalizeHist(gray)
                
                # Crop frame according to mask bounding box
                cropped = gray[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]]
                
                vol[z, :, :] = cropped
                
                z += 1
                            
            i += 1
        
        cap.release()
        
        output = sitk.GetImageFromArray(vol)   
        output.SetSpacing(self.voxel_spacing)

        self.volume = output
        
        print(f"Volume extraction completed. Final shape: {vol.shape}")
        return output

    