import argparse
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d


def visualize_stl_comparison(stl_path1: str, stl_path2: str):
    """
    Visualize two STL files overlaid to compare alignment.
    
    Args:
        stl_path1: Path to the first .stl file
        stl_path2: Path to the second .stl file
    """
    # Load meshes
    print(f"Loading: {stl_path1}")
    mesh1 = mesh.Mesh.from_file(stl_path1)
    
    print(f"Loading: {stl_path2}")
    mesh2 = mesh.Mesh.from_file(stl_path2)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add first mesh (red)
    collection1 = art3d.Poly3DCollection(mesh1.vectors, alpha=0.5)
    collection1.set_facecolor('red')
    collection1.set_edgecolor('darkred')
    collection1.set_linewidth(0.1)
    ax.add_collection3d(collection1)
    
    # Add second mesh (blue)
    collection2 = art3d.Poly3DCollection(mesh2.vectors, alpha=0.5)
    collection2.set_facecolor('blue')
    collection2.set_edgecolor('darkblue')
    collection2.set_linewidth(0.1)
    ax.add_collection3d(collection2)
    
    # Auto-scale axes
    all_points = np.vstack([
        mesh1.vectors.reshape(-1, 3),
        mesh2.vectors.reshape(-1, 3)
    ])
    
    max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2
    mid = (all_points.max(axis=0) + all_points.min(axis=0)) / 2
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('STL Comparison\nRed: {} | Blue: {}'.format(
        stl_path1.split('/')[-1], 
        stl_path2.split('/')[-1]
    ))
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and compare two STL files")
    parser.add_argument("stl1", help="Path to first .stl file")
    parser.add_argument("stl2", help="Path to second .stl file")
    
    args = parser.parse_args()
    visualize_stl_comparison(args.stl1, args.stl2)