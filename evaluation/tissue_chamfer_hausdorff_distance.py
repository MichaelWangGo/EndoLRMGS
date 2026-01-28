import open3d as o3d
import numpy as np
import cv2
import os
import json
from typing import List, Tuple
import re

def chamfer_distance(pcd1, pcd2):
    # Convert numpy arrays to Open3D point clouds
    pcd1_o3d = o3d.geometry.PointCloud()
    pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1)
    pcd2_o3d = o3d.geometry.PointCloud()
    pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2)

    # Compute nearest neighbor distances from pcd1 to pcd2
    dists1 = pcd1_o3d.compute_point_cloud_distance(pcd2_o3d)
    dists1 = np.asarray(dists1)

    # Compute nearest neighbor distances from pcd2 to pcd1
    dists2 = pcd2_o3d.compute_point_cloud_distance(pcd1_o3d)
    dists2 = np.asarray(dists2)

    # Chamfer distance is the sum of average distances in both directions
    chamfer_dist = np.mean(dists1) + np.mean(dists2)
    return chamfer_dist

def hausdorff_distance(pcd1, pcd2):
    # Convert numpy arrays to Open3D point clouds
    pcd1_o3d = o3d.geometry.PointCloud()
    pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1)
    pcd2_o3d = o3d.geometry.PointCloud()
    pcd2_o3d.points = o3d.utility.Vector3dVector(pcd2)

    # Compute nearest neighbor distances from pcd1 to pcd2
    dists1 = pcd1_o3d.compute_point_cloud_distance(pcd2_o3d)
    dists1 = np.asarray(dists1)

    # Compute nearest neighbor distances from pcd2 to pcd1
    dists2 = pcd2_o3d.compute_point_cloud_distance(pcd1_o3d)
    dists2 = np.asarray(dists2)

    # Hausdorff distance is the maximum of the maximum distances in both directions
    hausdorff_dist = max(np.max(dists1), np.max(dists2))
    return hausdorff_dist

def load_camera_intrinsics(json_path: str) -> Tuple[float, float, float, float]:
    with open(json_path, "r") as f:
        data = json.load(f)
    KL = data["camera-calibration"]["KL"]
    fx = KL[0][0]
    fy = KL[1][1]
    cx = KL[0][2]
    cy = KL[1][2]
    return fx, fy, cx, cy

def reconstruct_tools_from_depth_mask(depth_path, camera_intrinsics, mask_path=None):
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    depths = depth_image.astype(np.float32) #/255.0 # unit mm
    
    # If depth image has multiple channels, squeeze to 2D
    if len(depths.shape) == 3:
        depths = depths[:, :, 0]  # Take first channel or use np.squeeze if all channels are identical
    
    # Generate pixel coordinate grids
    height, width = depths.shape
    ys, xs = np.mgrid[0:height, 0:width]
    xs = xs.flatten()
    ys = ys.flatten()
    depths = depths.flatten()

    # Filter invalid depths
    valid = depths > 1
    
    # Apply mask filter if provided
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask from {mask_path}")
        mask_flat = mask.flatten()
        # import ipdb; ipdb.set_trace()
        valid = valid & (mask_flat == 255)
    
    xs, ys, depths = xs[valid], ys[valid], depths[valid]

    fx, fy, cx, cy = camera_intrinsics

    x3d = (xs - cx) * depths / fx
    y3d = (ys - cy) * depths / fy
    z3d = depths

    pts = np.vstack((x3d, y3d, z3d)).T

    return pts

def load_reference_point_cloud(pcd_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pcd.points)

def _extract_index(name: str) -> int:
    m = re.findall(r'\d+', name)
    return int(m[0]) if m else -1

def _collect_numbered(dir_path: str, exts) -> dict:
    out = {}
    for f in os.listdir(dir_path):
        if any(f.lower().endswith(e) for e in exts):
            idx = _extract_index(f)
            if idx >= 0:
                out[idx] = os.path.join(dir_path, f)
    return out

def find_numbered_matches(depth_dir: str, rgb_dir: str, pcd_dir: str) -> List[Tuple[str, str, str, str]]:
    depth_map = _collect_numbered(depth_dir, (".png", ".exr", ".tiff"))
    rgb_map   = _collect_numbered(rgb_dir, (".png", ".jpg"))
    pcd_map   = _collect_numbered(pcd_dir, (".ply", ".pcd", ".xyz"))
    common = sorted(set(depth_map) & set(rgb_map) & set(pcd_map))
    return [(depth_map[i], rgb_map[i], pcd_map[i]) for i in common]

def process_dataset(render_depth_dir, gt_depth_dir, mask_dir, intrinsics_json):
    fx, fy, cx, cy = load_camera_intrinsics(intrinsics_json)
    render_depths = _collect_numbered(render_depth_dir, (".png", ".exr", ".tiff"))
    gt_depths = _collect_numbered(gt_depth_dir, (".png", ".exr", ".tiff"))
    masks = _collect_numbered(mask_dir, (".png", ".jpg"))
    common = sorted(set(render_depths) & set(gt_depths) & set(masks))
    
    chamfers = []
    hausdorffs = []
    
    for idx in common:
        render_depth_path = render_depths[idx]
        gt_depth_path = gt_depths[idx]
        mask_path = masks[idx]
        
        render_pts = reconstruct_tools_from_depth_mask(render_depth_path, (fx, fy, cx, cy), mask_path)
        if render_pts.shape[0] == 0:
            continue
        
        gt_pts = reconstruct_tools_from_depth_mask(gt_depth_path, (fx, fy, cx, cy), mask_path)
        if gt_pts.shape[0] == 0:
            continue
        
        # Downsample the larger point cloud to match the smaller one
        if gt_pts.shape[0] > render_pts.shape[0]:
            sel = np.random.choice(gt_pts.shape[0], size=render_pts.shape[0], replace=False)
            gt_pts_ds = gt_pts[sel]
        else:
            gt_pts_ds = gt_pts

        cdist = chamfer_distance(render_pts, gt_pts_ds)
        hdist = hausdorff_distance(render_pts, gt_pts_ds)
        chamfers.append(cdist)
        hausdorffs.append(hdist)
        print(f"{os.path.basename(render_depth_path)} Chamfer={cdist:.6f} Hausdorff={hdist:.6f}")
    
    if chamfers:
        print(f"Average Chamfer: {np.mean(chamfers):.6f}")
    if hausdorffs:
        print(f"Average Hausdorff: {np.mean(hausdorffs):.6f}")

def main():
    render_depth_dir = "/workspace/ForPlane/exps/endonerf/pulling/endonerf_32k/estm/test_rendered_depth"
    gt_depth_dir = "/workspace/ForPlane/exps/endonerf/pulling/endonerf_32k/estm/processed_gt_depth"
    mask_dir = "/workspace/ForPlane/exps/endonerf/pulling/endonerf_32k/estm/processed_mask"

    camera_intrinsics_json = "/workspace/datasets/endolrm_dataset/endonerf/pulling/frame_data.json"
    process_dataset(render_depth_dir, gt_depth_dir, mask_dir, camera_intrinsics_json)

if __name__ == "__main__":
    main()