import os
import argparse
import numpy as np
import SimpleITK as sitk
from skimage import measure
from stl import mesh


def nifti_to_stl(input_path: str, output_path: str, threshold: float = 0.5, smooth: bool = True):
    """
    Convert a NIfTI (.nii.gz) file to an STL mesh file.
    
    Args:
        input_path: Path to the input .nii.gz file
        output_path: Path to the output .stl file
        threshold: Threshold value for marching cubes (default 0.5 for binary masks)
        smooth: Whether to apply smoothing to the mesh
    """
    # Read the NIfTI file
    print(f"Reading: {input_path}")
    img = sitk.ReadImage(input_path)
    
    # Get the spacing for proper mesh scaling
    spacing = img.GetSpacing()
    
    # Convert to numpy array
    volume = sitk.GetArrayFromImage(img)
    
    # Apply marching cubes to generate mesh
    print("Generating mesh with marching cubes...")
    verts, faces, normals, values = measure.marching_cubes(
        volume, 
        level=threshold,
        spacing=spacing[::-1],  # Reverse spacing to match numpy array order (z, y, x)
        step_size=1
    )
    
    print(f"Mesh generated: {len(verts)} vertices, {len(faces)} faces")
    
    # Create the STL mesh
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    for i, face in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[face[j], :]
    
    # Save the mesh
    print(f"Saving STL to: {output_path}")
    stl_mesh.save(output_path)
    print("Done!")


def convert_directory(input_dir: str, output_dir: str, threshold: float = 0.5):
    """
    Convert all .nii.gz files in a directory to .stl files.
    
    Args:
        input_dir: Directory containing .nii.gz files
        output_dir: Directory to save .stl files
        threshold: Threshold value for marching cubes
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.nii.gz'):
            input_path = os.path.join(input_dir, filename)
            output_filename = filename.replace('.nii.gz', '.stl')
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                nifti_to_stl(input_path, output_path, threshold)
            except Exception as e:
                print(f"Error converting {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NIfTI files to STL meshes")
    parser.add_argument("input", help="Input .nii.gz file or directory")
    parser.add_argument("-o", "--output", help="Output .stl file or directory")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, 
                        help="Threshold for marching cubes (default: 0.5)")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        output_dir = args.output or os.path.join(args.input, "stl_output")
        convert_directory(args.input, output_dir, args.threshold)
    else:
        output_path = args.output or args.input.replace('.nii.gz', '.stl')
        nifti_to_stl(args.input, output_path, args.threshold)