import numpy as np
import cv2
import json
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import math

def load_calibration(calib_file):
    """Load camera calibration parameters from JSON file"""
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    
    camera_matrix = np.array(calib['K1']['data']).reshape(3, 3).astype(np.float32)
    image_size = tuple(map(int, calib['image_size']['data']))
    
    return camera_matrix, image_size

def project_pointcloud(point_cloud_path, camera_matrix, image_size, output_path=None):
    """Project point cloud to 2D image using camera intrinsics"""
    # Load point cloud
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    points = np.asarray(pcd.points)  # (N, 3)
    colors = np.asarray(pcd.colors)  # (N, 3)

    # Project points to image plane
    points_camera = points.T  # (3, N)
    projections = camera_matrix @ points_camera  # (3, N)
    projections /= projections[2, :]  # Normalize by depth
    
    # Get pixel coordinates
    pixels = projections[:2, :].T  # (N, 2)
    
    # Create image
    h, w = image_size
    # image = np.zeros((h, w, 3), dtype=np.uint8)
    image  = np.ones((math.ceil(pixels[:, 1].max()), math.ceil(pixels[:, 0].max()), 3), dtype=np.uint8) * 255
    
    # Filter valid pixels
    # valid_mask = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & \
    #             (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    
    # valid_mask = (pixels[:, 0] < w) &  (pixels[:, 1] < h)
    
    # valid_pixels = pixels[valid_mask].astype(int)
    # valid_colors = (colors[valid_mask] * 255).astype(np.uint8)
    # import ipdb; ipdb.set_trace()
    valid_pixels = pixels.astype(int)
    valid_colors = (colors * 255).astype(np.uint8)
    
    # Paint pixels
    for (x, y), color in zip(valid_pixels, valid_colors):
        image[y, x] = color

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    return image

def main():
    # Paths
    pointcloud_dir = Path('/workspace/EndoLRM/final_results_tissue_tools/endonerf/pulling_masked')
    output_dir = pointcloud_dir / 'reprojected_images'
    calib_file = '/workspace/dataset/endolrm_dataset/endonerf/calib.json'
    output_dir.mkdir(exist_ok=True)
    
    # Load calibration
    camera_matrix, image_size = load_calibration(calib_file)
    print("Loaded camera calibration:")
    print(f"Camera Matrix:\n{camera_matrix}")
    print(f"Image size: {image_size}")
    
    # Process all point clouds
    point_clouds = sorted(pointcloud_dir.glob('*.ply'))
    
    for pcd_path in tqdm(point_clouds):
        output_path = output_dir / f"{pcd_path.stem}_reprojected.png"
        try:
            project_pointcloud(pcd_path, camera_matrix, image_size, str(output_path))
            print(f"Processed {pcd_path.name} -> {output_path.name}")
        except Exception as e:
            print(f"Error processing {pcd_path.name}: {str(e)}")
            continue

if __name__ == '__main__':
    main()