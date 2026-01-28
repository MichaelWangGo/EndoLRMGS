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

def reconstruct_tools_from_depth_mask(depth_path, camera_intrinsics):
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    depths = depth_image.astype(np.float32) #/255.0 # unit mm

    # Generate pixel coordinate grids
    height, width = depth_image.shape
    ys, xs = np.mgrid[0:height, 0:width]
    xs = xs.flatten()
    ys = ys.flatten()
    depths = depths.flatten()

    # Filter invalid depths
    valid = depths > 0
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

def process_dataset(pcd_dir, depth_dir, rgb_dir, intrinsics_json):
    fx, fy, cx, cy = load_camera_intrinsics(intrinsics_json)
    pairs = find_numbered_matches(depth_dir, rgb_dir, pcd_dir)
    chamfers = []
    hausdorffs = []
    # saved_sample = False
    for depth_path, rgb_path, ref_path in pairs:
        # Load rgb just to verify; ignore if missing
        _ = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        recon_pts = reconstruct_tools_from_depth_mask(depth_path, (fx, fy, cx, cy))
        if recon_pts.shape[0] == 0:
            continue
        
        # # Save first valid reconstruction as sample
        # if not saved_sample:
        #     sample_pcd = o3d.geometry.PointCloud()
        #     sample_pcd.points = o3d.utility.Vector3dVector(recon_pts)
        #     sample_path = "/workspace/EndoLRMGS/stereomis/p1/reconstructed_tool_sample.ply"
        #     o3d.io.write_point_cloud(sample_path, sample_pcd)
        #     print(f"Saved sample reconstructed tool to {sample_path}")
        #     saved_sample = True
        
        ref_pts = load_reference_point_cloud(ref_path)
        if ref_pts.shape[0] == 0:
            continue
        # Downsample reference point cloud to match reconstruction size
        if ref_pts.shape[0] > recon_pts.shape[0]:
            sel = np.random.choice(ref_pts.shape[0], size=recon_pts.shape[0], replace=False)
            ref_pts_ds = ref_pts[sel]
        else:
            ref_pts_ds = ref_pts
        cdist = chamfer_distance(recon_pts, ref_pts_ds)
        hdist = hausdorff_distance(recon_pts, ref_pts_ds)
        chamfers.append(cdist)
        hausdorffs.append(hdist)
        print(f"{os.path.basename(depth_path)} -> {os.path.basename(ref_path)} Chamfer={cdist:.6f} Hausdorff={hdist:.6f}")
    if chamfers:
        print(f"Average Chamfer: {np.mean(chamfers):.6f}")
    if hausdorffs:
        print(f"Average Hausdorff: {np.mean(hausdorffs):.6f}")

def main():
    recon_pcd_dir = "/workspace/EndoLRMGS/stereomis/p1/zxhezexin/openlrm-mix-base-1.1/tissue_reconstruction"
    gt_depth_dir = "/workspace/datasets/endolrm_dataset/stereomis/p1/depth"
    gt_rgb_dir = "/workspace/datasets/endolrm_dataset/stereomis/p1/left_finalpass"
    camera_intrinsics_json = "/workspace/datasets/endolrm_dataset/stereomis/p1/frame_data.json"
    process_dataset(recon_pcd_dir, gt_depth_dir, gt_rgb_dir, camera_intrinsics_json)

if __name__ == "__main__":
    main()