import open3d as o3d
import os
import numpy as np

def combine_pointclouds(tools_file, tissue_file, output_file):
    """Combine tools and tissue pointclouds by concatenating their points and colors"""
    # Load pointclouds
    tools_pcd = o3d.io.read_point_cloud(tools_file)
    tissue_pcd = o3d.io.read_point_cloud(tissue_file)
    
    # Get points and colors from both pointclouds
    tools_points = np.asarray(tools_pcd.points)
    tools_colors = np.asarray(tools_pcd.colors)
    tissue_points = np.asarray(tissue_pcd.points)
    tissue_colors = np.asarray(tissue_pcd.colors)
    
    # Concatenate points and colors
    combined_points = np.concatenate([tools_points, tissue_points], axis=0)
    combined_colors = np.concatenate([tools_colors, tissue_colors], axis=0)
    
    # Create new pointcloud with combined data
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    
    # Save combined pointcloud
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    o3d.io.write_point_cloud(output_file, combined_pcd)
    
    return combined_pcd

def process_sequence(tools_path, tissue_path, output_path):
    """Process all matching pairs of pointclouds by their order in sorted lists"""
    # Get sorted lists of files
    tools_files = sorted([f for f in os.listdir(tools_path) if f.endswith(".ply")])
    tissue_files = sorted([f for f in os.listdir(tissue_path) if f.endswith(".ply")])
    
    # Get the minimum length to avoid index out of range
    n_files = min(len(tools_files), len(tissue_files))
    
    for i in range(n_files):
        tools_file = tools_files[i]
        tissue_file = tissue_files[i]
        
        tools_full_path = os.path.join(tools_path, tools_file)
        tissue_full_path = os.path.join(tissue_path, tissue_file)
        output_file = os.path.join(output_path, f"combined_{i:04d}.ply")
        
        print(f"\nProcessing pair {i+1}/{n_files}")
        print(f"Tools: {tools_file}")
        print(f"Tissue: {tissue_file}")
        
        try:
            combined_pcd = combine_pointclouds(tools_full_path, tissue_full_path, output_file)
            print(f"Saved combined pointcloud to: {output_file}")
            print(f"Combined pointcloud has {len(combined_pcd.points)} points")
        except Exception as e:
            print(f"Error processing pair {i+1}: {str(e)}")
            continue

if __name__ == "__main__":
    tools_path = "/workspace/EndoLRMGS/endonerf/pulling/postprocessed_tools"
    tissue_path = "/workspace/EndoLRMGS/endonerf/pulling/tissue_reconstruction"
    output_path = "/workspace/EndoLRMGS/endonerf/pulling/final_results"
    
    process_sequence(tools_path, tissue_path, output_path)