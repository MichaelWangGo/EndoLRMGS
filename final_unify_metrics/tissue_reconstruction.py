import numpy as np
import cv2
import open3d as o3d
import json
from pathlib import Path



def load_camera_params(calib_file):
    """Load camera parameters from frame_data.json."""
    with open(calib_file, 'r') as f:
        calib = json.load(f)
    cam_calib = calib.get("camera-calibration", {})
    # Use left camera intrinsics/distortion
    K = np.array(cam_calib["KL"]).reshape(3, 3)
    D = np.array(cam_calib["DL"]).reshape(-1)
    return K, D

def create_point_cloud(rgb_file, depth_file, K, D, save_path=None):
    """Create point cloud from RGB and depth images."""
    # Read images
    rgb = cv2.imread(rgb_file)
    rgb = cv2.resize(rgb, (1280, 1024))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE).astype(float)
    depth = cv2.resize(depth, (1280, 1024))
    # depth = depth[:,:,0] 
    # import ipdb; ipdb.set_trace()
    # Undistort images
    h, w = depth.shape
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    rgb = cv2.undistort(rgb, K, D, None, newcameramtx)
    depth = cv2.undistort(depth, K, D, None, newcameramtx)

    # Detect black regions (where all RGB channels are very dark)
    black_threshold = 10  # Adjust this value if needed
    not_black_mask = (rgb.mean(axis=2) > black_threshold)

    # Create point cloud
    fx, fy = newcameramtx[0,0], newcameramtx[1,1]
    cx, cy = newcameramtx[0,2], newcameramtx[1,2]
    
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Calculate 3D points
    z = depth #/ 255 # if reconstruct from rendered images, depth values should'nt be divided by 255; if from real depth images, should be divided by 255
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    # Stack points and colors
    xyz = np.stack([x, y, z], axis=-1)
    rgb = rgb.astype(float) / 255.0

    # Create and save point cloud, excluding black regions
    pcd = o3d.geometry.PointCloud()
    valid_mask = (~np.isnan(xyz).any(axis=-1) & 
                 (z > 0) & 
                 not_black_mask)  # Add the non-black mask
    pcd.points = o3d.utility.Vector3dVector(xyz[valid_mask])
    pcd.colors = o3d.utility.Vector3dVector(rgb[valid_mask])

    if save_path:
        o3d.io.write_point_cloud(save_path, pcd)
    
    return pcd

def main():
    # Create output directory
    output_dir = Path("/workspace/EndoLRMGS/stereomis/p3/zxhezexin/openlrm-mix-base-1.1/tissue_reconstruction")
    output_dir.mkdir(exist_ok=False, parents=True)

    rgb_path = "/workspace/EndoLRMGS/stereomis/p3/zxhezexin/openlrm-mix-base-1.1/rendered"
    depth_path = "/workspace/EndoLRMGS/stereomis/p3/zxhezexin/openlrm-mix-base-1.1/rendered_depth"
    calib_file = "/workspace/datasets/endolrm_dataset/stereomis/p3/frame_data.json"

    # Load camera parameters from calibration file
    K, D = load_camera_params(calib_file)
    print("Loaded camera parameters (from frame_data.json):")
    print(f"Intrinsic matrix K:\n{K}")
    print(f"Distortion coefficients D: {D}")

    # Process all images
    rgb_files = sorted(Path(rgb_path).glob("*.png"))
    depth_files = sorted(Path(depth_path).glob("*.png"))

    print(f"Found {len(rgb_files)} RGB images and {len(depth_files)} depth maps")
    
    for i, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        print(f"Processing frame {i+1}/{len(rgb_files)}")
        
        output_file = output_dir / f"frame_{i:04d}.ply"
        try:
            pcd = create_point_cloud(
                str(rgb_file),
                str(depth_file),
                K, D,
                save_path=str(output_file)
            )
            print(f"Saved point cloud to {output_file}")
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

    print("Point cloud reconstruction completed!")

if __name__ == "__main__":
    main()