import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import ConvexHull
import os
import glob
import pandas as pd

# Add CUDA support
try:
    import cupy as cp
    CUDA_AVAILABLE = True
    # Set CUDA device
    cp.cuda.Device(0).use()
    print("CUDA acceleration enabled on NVIDIA RTX A6000")
except ImportError:
    CUDA_AVAILABLE = False
    print("CuPy not available, using CPU")

def perspective_project(points, K, use_cuda=True):
    """Project 3D points to 2D using perspective projection"""
    if CUDA_AVAILABLE and use_cuda and isinstance(points, np.ndarray):
        points_gpu = cp.asarray(points)
        K_gpu = cp.asarray(K)
        points_2d = K_gpu @ points_gpu.T
        points_2d = points_2d / points_2d[2]
        return cp.asnumpy(points_2d[:2].T)
    else:
        points_2d = K @ points.T
        points_2d = points_2d / points_2d[2]
        return points_2d[:2].T

def transform_points(points, T_source, T_target, use_cuda=True):
    """Transform points from source camera frame to target camera frame"""
    if CUDA_AVAILABLE and use_cuda:
        points_gpu = cp.asarray(points)
        # Convert to homogeneous coordinates
        points_homo = cp.concatenate([points_gpu, cp.ones((points_gpu.shape[0], 1))], axis=1)
        
        # Compute transformation matrix from source to target
        T_source_gpu = cp.asarray(T_source)
        T_target_gpu = cp.asarray(T_target)
        T_source_inv = cp.linalg.inv(T_source_gpu)
        T_transform = T_target_gpu @ T_source_inv
        
        # Transform points
        points_transformed = (T_transform @ points_homo.T).T
        return cp.asnumpy(points_transformed[:, :3])
    else:
        # Convert to homogeneous coordinates
        points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        # Compute transformation matrix from source to target
        T_source_inv = np.linalg.inv(T_source)
        T_transform = T_target @ T_source_inv
    
        # Transform points
        points_transformed = (T_transform @ points_homo.T).T
        return points_transformed[:, :3]

def calculate_orthographic_matrix(points, use_cuda=True):
    """Calculate orthographic projection matrix based on point bounds"""
    if CUDA_AVAILABLE and use_cuda:
        points_gpu = cp.asarray(points)
        l = float(cp.min(points_gpu[:, 0]))
        r = float(cp.max(points_gpu[:, 0]))
        b = float(cp.min(points_gpu[:, 1]))
        t = float(cp.max(points_gpu[:, 1]))
        n = float(cp.min(points_gpu[:, 2]))
        f = float(cp.max(points_gpu[:, 2]))
    else:
        l = np.min(points[:, 0])
        r = np.max(points[:, 0])
        b = np.min(points[:, 1])
        t = np.max(points[:, 1])
        n = np.min(points[:, 2])
        f = np.max(points[:, 2])
    
    return np.array([
        [2.0/(r-l), 0, 0, 0],
        [0, 2.0/(t-b), 0, 0],
        [0, 0, -2.0/(f-n), 0],
        [-(r+l)/(r-l), -(t+b)/(t-b), -(f+n)/(f-n), 1]
    ])

def calculate_scale_factor(bg_points, tool_points, ortho_matrix, use_cuda=True):
    """Calculate scale factor between background and tool point clouds"""
    if CUDA_AVAILABLE and use_cuda:
        bg_points_gpu = cp.asarray(bg_points)
        tool_points_gpu = cp.asarray(tool_points)
        ortho_matrix_gpu = cp.asarray(ortho_matrix)
        
        # Project background points
        bg_homogeneous = cp.concatenate([bg_points_gpu, cp.ones((bg_points_gpu.shape[0], 1))], axis=1)
        bg_ortho = (ortho_matrix_gpu @ bg_homogeneous.T).T[:, :2]
        
        # Project tool points
        tool_homogeneous = cp.concatenate([tool_points_gpu, cp.ones((tool_points_gpu.shape[0], 1))], axis=1)
        tool_ortho = (ortho_matrix_gpu @ tool_homogeneous.T).T[:, :2]
        
        bg_area0 = float((cp.max(bg_ortho[:, 0]) - cp.min(bg_ortho[:, 0])) * (cp.max(bg_ortho[:, 1]) - cp.min(bg_ortho[:, 1])))
        tool_area0 = float((cp.max(tool_ortho[:, 0]) - cp.min(tool_ortho[:, 0])) * (cp.max(tool_ortho[:, 1]) - cp.min(tool_ortho[:, 1])))
        
        # Convert back to CPU for convex hull calculation
        bg_ortho_cpu = cp.asnumpy(bg_ortho)
        tool_ortho_cpu = cp.asnumpy(tool_ortho)
    else:
        # Project background points
        bg_homogeneous = np.concatenate([bg_points, np.ones((bg_points.shape[0], 1))], axis=1)
        bg_ortho = (ortho_matrix @ bg_homogeneous.T).T[:, :2]
        
        # Project tool points
        tool_homogeneous = np.concatenate([tool_points, np.ones((tool_points.shape[0], 1))], axis=1)
        tool_ortho = (ortho_matrix @ tool_homogeneous.T).T[:, :2]
        bg_area0 = (np.max(bg_ortho[:, 0]) - np.min(bg_ortho[:, 0])) * (np.max(bg_ortho[:, 1]) - np.min(bg_ortho[:, 1]))
        tool_area0 = (np.max(tool_ortho[:, 0]) - np.min(tool_ortho[:, 0])) * (np.max(tool_ortho[:, 1]) - np.min(tool_ortho[:, 1]))
        bg_ortho_cpu = bg_ortho
        tool_ortho_cpu = tool_ortho
        
    # Calculate areas using convex hull
    bg_area1 = calculate_precise_area(bg_ortho_cpu)
    tool_area1 = calculate_precise_area(tool_ortho_cpu)
    scale_factor0 = np.sqrt(bg_area0 / tool_area0)
    scale_factor1 = np.sqrt(bg_area1 / tool_area1)
    scale_factor = (scale_factor0 + scale_factor1) / 2

    return scale_factor, bg_ortho_cpu, tool_ortho_cpu

def calculate_precise_area(points_2d):
    """
    Calculate area using convex hull of 2D points
    Args:
        points_2d: Nx2 array of 2D points
    Returns:
        area: float, area of the convex hull
    """
    if len(points_2d) < 3:
        return 0
    try:
        hull = ConvexHull(points_2d)
        return hull.area
    except Exception as e:
        print(f"Warning: Convex hull calculation failed: {e}")
        return 0

# Create background point clouds for each masked region
def create_masked_pointcloud(masked_depth, K, use_cuda=True):
    """Create point cloud from masked depth using CUDA acceleration"""
    height, width = masked_depth.shape
    
    if CUDA_AVAILABLE and use_cuda:
        x = cp.arange(width)
        y = cp.arange(height)
        xx, yy = cp.meshgrid(x, y)
        
        # Create homogeneous pixel coordinates
        pixels = cp.stack([xx, yy, cp.ones_like(xx)], axis=-1)
        K_gpu = cp.asarray(K)
        K_inv = cp.linalg.inv(K_gpu)
        pixels_normalized = K_inv @ pixels.reshape(-1, 3).T
        
        masked_depth_gpu = cp.asarray(masked_depth)
        points_3d = pixels_normalized.T * masked_depth_gpu.reshape(-1, 1)
        
        # Remove zero points
        valid_mask = cp.any(points_3d != 0, axis=1)
        valid_points = cp.asnumpy(points_3d[valid_mask])
    else:
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        
        # Create homogeneous pixel coordinates
        pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
        pixels_normalized = np.linalg.inv(K) @ pixels.reshape(-1, 3).T
        points_3d = pixels_normalized.T * masked_depth.reshape(-1, 1)
        # Remove zero points
        valid_points = points_3d[np.any(points_3d != 0, axis=1)]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    return pcd, valid_points

def process_masked_region(mask, depth, K, use_cuda=True):
    """Process a masked region to create perspective point cloud"""
    y_coords, x_coords = np.where(mask == 255)
    
    if CUDA_AVAILABLE and use_cuda:
        x_coords_gpu = cp.asarray(x_coords)
        y_coords_gpu = cp.asarray(y_coords)
        pixels = cp.stack([x_coords_gpu, y_coords_gpu, cp.ones_like(x_coords_gpu)], axis=-1)
        depths = cp.asarray(depth[y_coords, x_coords])
        
        # Convert to 3D points using perspective projection
        K_gpu = cp.asarray(K)
        K_inv = cp.linalg.inv(K_gpu)
        pixels_normalized = K_inv @ pixels.T
        points_3d = pixels_normalized.T * depths[:, cp.newaxis]
        points_3d = cp.asnumpy(points_3d)
    else:
        pixels = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
        depths = depth[y_coords, x_coords]
        
        # Convert to 3D points using perspective projection
        pixels_normalized = np.linalg.inv(K) @ pixels.T
        points_3d = pixels_normalized.T * depths[:, np.newaxis]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    return pcd, points_3d


def get_pointcloud_pairs(pcd_dir):
    """Get all groups of point clouds (up to 3 masks) from directory"""
    # Define possible patterns for mask0 files
    patterns = [
        "*_mask0.ply",
        "*.color_mask0.ply"
    ]
    
    groups = []
    for pattern in patterns:
        pcd0_files = sorted(glob.glob(os.path.join(pcd_dir, pattern)))
        
        for pcd0_file in pcd0_files:
            # Handle different naming formats
            if "stereomis" in pcd0_file:
                base_path = pcd0_file.replace("_mask0.ply", "")
                frame_num = os.path.basename(pcd0_file).split("_")[1]
                mask_files = [pcd0_file]
                
                # Check for mask1 and mask2
                for mask_idx in [1, 2]:
                    mask_file = f"{base_path}_mask{mask_idx}.ply"
                    if os.path.exists(mask_file):
                        mask_files.append(mask_file)
                
            elif "endonerf" in pcd0_file:
                base_path = pcd0_file.replace(".color_mask0.ply", "")
                frame_num = os.path.basename(pcd0_file).split("-")[1].split(".")[0]
                mask_files = [pcd0_file]
                
                # Check for mask1 and mask2
                for mask_idx in [1, 2]:
                    mask_file = f"{base_path}.color_mask{mask_idx}.ply"
                    if os.path.exists(mask_file):
                        mask_files.append(mask_file)
            
            # Only add if we have at least 2 masks
            if len(mask_files) >= 2:
                groups.append((frame_num, mask_files))
    
    return sorted(groups)


def get_z_at_xy(bg_points, target_x, target_y):
    # Find points that match the x,y coordinates 
    matches = bg_points[
        (bg_points[:, 0] == target_x) & 
        (bg_points[:, 1] == target_y)
    ]
    
    if len(matches) > 0:
        # Return the z value of the first matching point as int
        return int(matches[0][2])
    else:
        # If no exact match, find nearest point
        distances = np.sqrt(
            (bg_points[:, 0] - target_x)**2 + (bg_points[:, 1] - target_y)**2
        )
        nearest_idx = np.argmin(distances)
        return int(bg_points[nearest_idx][2])


def process_sequence(pcd_paths, depth_path, mask_path, output_path, frame_num):
    """Process a sequence of 2 or 3 point clouds with transformations and scaling."""
    # Load point clouds
    pcd_list = [o3d.io.read_point_cloud(p) for p in pcd_paths]
    # Load mask and depth
    mask_rgb = cv2.imread(mask_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Unique tool colors (exclude black)
    unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]
    n_tools = min(len(pcd_list), len(unique_colors))
    if n_tools < 2:
        raise ValueError(f"Need ≥2 tools, got {n_tools}")

    # Build binary masks per tool
    tool_masks = [
        (np.all(mask_rgb == unique_colors[i], axis=2).astype(np.uint8) * 255)
        for i in range(n_tools)
    ]

    transformed_pcds = []
    results = []
    initial_iou_scores = []
    final_iou_scores = []
    initial_projections = []
    initial_transformed_pcds = []

    for tool_idx in range(n_tools):
        pcd = pcd_list[tool_idx]
        mask = tool_masks[tool_idx]

        # 1. Transform points
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        transformed_points = transform_points(points, T_source, T_target, use_cuda=CUDA_AVAILABLE)

        transformed_pcd_initial = o3d.geometry.PointCloud()
        transformed_pcd_initial.points = o3d.utility.Vector3dVector(transformed_points)
        transformed_pcd_initial.colors = o3d.utility.Vector3dVector(colors)
        initial_transformed_pcds.append(transformed_pcd_initial)

        # 2. Background points from mask & depth
        _, bg_points = process_masked_region(mask, depth, K1, use_cuda=CUDA_AVAILABLE)

        # 3. Ortho matrix & scale
        ortho_matrix = calculate_orthographic_matrix(bg_points, use_cuda=CUDA_AVAILABLE)
        scale_factor, _, _ = calculate_scale_factor(bg_points, transformed_points, ortho_matrix, use_cuda=CUDA_AVAILABLE)
        scaled_points = transformed_points * scale_factor

        # Initial IoU
        height, width = mask.shape
        mask_binary = (mask == 255)
        proj_pts_init = perspective_project(scaled_points, K1, use_cuda=CUDA_AVAILABLE)
        init_mask = np.zeros((height, width), dtype=bool)
        valid = (
            (proj_pts_init[:, 0] >= 0) & (proj_pts_init[:, 0] < width) &
            (proj_pts_init[:, 1] >= 0) & (proj_pts_init[:, 1] < height)
        )
        if np.any(valid):
            coords = proj_pts_init[valid].astype(int)
            init_mask[coords[:, 1], coords[:, 0]] = True
            init_mask = cv2.dilate(init_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            inter = np.sum(init_mask & mask_binary)
            union = np.sum(init_mask | mask_binary)
            initial_iou = inter / union if union > 0 else 0.0
        else:
            initial_iou = 0.0
        initial_iou_scores.append(initial_iou)
        initial_projections.append(init_mask)

        # Optimization search ranges
        pts_range = np.ptp(scaled_points, axis=0)
        max_z_idx = np.argmax(scaled_points[:, 2])
        max_point = scaled_points[max_z_idx]
        x_coord, y_coord, z_coord_tools = max_point
        # Match depth (fallback nearest if exact not found)
        match_z = get_z_at_xy(bg_points, x_coord, y_coord)
        z_range = match_z - z_coord_tools

        x_translations = np.linspace(-pts_range[0] / 2, pts_range[0] / 2, 10)
        y_translations = np.linspace(-pts_range[1] / 2, pts_range[1] / 2, 10)
        z_translations = np.linspace(0, z_range, 20)

        best_score = float('inf')
        best_transform = np.zeros(3, dtype=np.float32)
        best_projection = np.zeros((height, width), dtype=bool)
        best_iou = 0.0

        if CUDA_AVAILABLE:
            tool_pts_gpu = cp.asarray(scaled_points)
            K_gpu = cp.asarray(K1)
            mask_bin_gpu = cp.asarray(mask_binary)
            for dx in x_translations:
                for dy in y_translations:
                    for dz in z_translations:
                        translated = tool_pts_gpu + cp.array([dx, dy, dz])
                        proj = K_gpu @ translated.T
                        proj = proj / proj[2]
                        proj2d = cp.asnumpy(proj[:2].T)
                        pmask = np.zeros((height, width), dtype=bool)
                        v = (
                            (proj2d[:, 0] >= 0) & (proj2d[:, 0] < width) &
                            (proj2d[:, 1] >= 0) & (proj2d[:, 1] < height)
                        )
                        if not np.any(v):
                            continue
                        c = proj2d[v].astype(int)
                        pmask[c[:, 1], c[:, 0]] = True
                        pmask = cv2.dilate(pmask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                        inter = np.sum(pmask & mask_binary)
                        union = np.sum(pmask | mask_binary)
                        iou = inter / union if union > 0 else 0.0
                        score = 1.0 - iou
                        if score < best_score:
                            best_score = score
                            best_transform = np.array([dx, dy, dz])
                            best_projection = pmask
                            best_iou = iou
        else:
            for dx in x_translations:
                for dy in y_translations:
                    for dz in z_translations:
                        translated = scaled_points + np.array([dx, dy, dz])
                        proj2d = perspective_project(translated, K1, use_cuda=False)
                        pmask = np.zeros((height, width), dtype=bool)
                        v = (
                            (proj2d[:, 0] >= 0) & (proj2d[:, 0] < width) &
                            (proj2d[:, 1] >= 0) & (proj2d[:, 1] < height)
                        )
                        if not np.any(v):
                            continue
                        c = proj2d[v].astype(int)
                        pmask[c[:, 1], c[:, 0]] = True
                        pmask = cv2.dilate(pmask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                        inter = np.sum(pmask & mask_binary)
                        union = np.sum(pmask | mask_binary)
                        iou = inter / union if union > 0 else 0.0
                        score = 1.0 - iou
                        if score < best_score:
                            best_score = score
                            best_transform = np.array([dx, dy, dz])
                            best_projection = pmask
                            best_iou = iou

        final_points = scaled_points + best_transform
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(final_points)
        transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
        transformed_pcds.append(transformed_pcd)
        results.append((best_transform[0], best_transform[1], best_transform[2], best_projection))
        final_iou_scores.append(best_iou)

        print(f"Tool {tool_idx} - Initial IoU: {initial_iou:.4f} | Final IoU: {best_iou:.4f}")

    # Save combined initial point clouds
    initial_tools_dir = os.path.join(output_base_dir.replace('postprocessed_tools', 'initial_tools'))
    os.makedirs(initial_tools_dir, exist_ok=True)
    combined_initial = initial_transformed_pcds[0]
    for extra in initial_transformed_pcds[1:]:
        combined_initial = combined_initial + extra
    # o3d.io.write_point_cloud(os.path.join(initial_tools_dir, f"frame_{frame_num}_combined_initial.ply"), combined_initial)

    # Initial projection visualization
    height, width = tool_masks[0].shape
    if len(initial_projections) >= 2:
        init_vis = np.zeros((height, width, 3), dtype=np.uint8)
        # Ground truth in red
        for m in tool_masks:
            init_vis[m == 255] = [255, 0, 0]
        # Projections in green; overlaps yellow
        for i, proj in enumerate(initial_projections):
            init_vis[proj] = [0, 255, 0]
            init_vis[(tool_masks[i] == 255) & proj] = [255, 255, 0]
        cv2.imwrite(os.path.join(output_base_dir, f"projection_comparison_initial{frame_num}.png"), init_vis)

    # Final projection visualization (use all masks and projections)
    if len(results) >= 1:
        final_vis = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw all GT masks in red
        for m in tool_masks:
            final_vis[m == 255] = [255, 0, 0]
        # Draw all projections in green; overlaps become yellow
        for i, res in enumerate(results):
            proj = res[3]
            final_vis[proj] = [0, 255, 0]
            final_vis[(tool_masks[i] == 255) & proj] = [255, 255, 0]
        cv2.imwrite(os.path.join(output_base_dir, f"projection_comparison_final{frame_num}.png"), final_vis)

    # Combined final cloud
    combined_final = transformed_pcds[0]
    for extra in transformed_pcds[1:]:
        combined_final = combined_final + extra
    o3d.io.write_point_cloud(output_path, combined_final)

    return transformed_pcds, initial_iou_scores, final_iou_scores

# Define camera matrices and parameters
T_source = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, -2],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

T_target = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

# K1 = np.array([
#     [1033.894287109375, 0.0, 604.578857421875],
#     [0.0, 1033.7147216796875, 514.9761962890625],
#     [0.0, 0.0, 1.0]
# ]) # p2_0 p2_8

K1 = np.array([
    [1039.6275634765625, 0.0, 596.5435180664062],
    [0.0, 1039.4129638671875, 502.235595703125],
    [0.0, 0.0, 1.0]
]) # p1

# K1 = np.array([
#     [1042.4930419921875, 0.0, 608.7192993164062],
#     [0.0, 1042.2545166015625, 489.70538330078125],
#     [0.0, 0.0, 1.0]
# ]) # p2_3

# K1 = np.array([
#     [1042.4930419921875, 0.0, 608.7192993164062],
#     [0.0, 1042.2545166015625, 489.70538330078125],
#     [0.0, 0.0, 1.0]
# ]) # p3


if __name__ == "__main__":
    # # Define base directories
    pcd_base_dir = "/workspace/EndoLRMGS/stereomis/p1_new/zxhezexin/openlrm-mix-base-1.1/meshes"
    depth_base_dir = "/workspace/EndoLRMGS/stereomis/p1_new/zxhezexin/openlrm-mix-base-1.1/rendered_depth"
    mask_base_dir = "/workspace/datasets/endolrm_dataset/stereomis/p1_new/Annotations"
    output_base_dir = "/workspace/EndoLRMGS/stereomis/p1_new/postprocessed_tools"

    # Create output folder if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all pairs of point clouds
    pointcloud_pairs = get_pointcloud_pairs(pcd_base_dir)

    # Create lists to store results
    all_results = []
    
    # Process each pair
    for frame_num, pcd_paths in pointcloud_pairs:
        depth_path = os.path.join(depth_base_dir, f"frame_{frame_num}_endo_depth.png")
        # depth_path = os.path.join(depth_base_dir, f"frame_{frame_num}.png")
        mask_path = os.path.join(mask_base_dir, f"frame_{frame_num}.png")
        output_path = os.path.join(output_base_dir, f"frame_{frame_num}_combined.ply")
        if not (os.path.exists(depth_path) and os.path.exists(mask_path)):
            print(f"Missing depth or mask file for frame {frame_num}")
            continue
        if len(pcd_paths) < 2:
            print(f"Skipping frame {frame_num}, only {len(pcd_paths)} point clouds")
            continue
        print(f"Processing frame {frame_num} with {len(pcd_paths)} tools...")
        try:
            final_pcds, initial_ious, final_ious = process_sequence(pcd_paths, depth_path, mask_path, output_path, frame_num)
            result_row = {'frame': frame_num}
            for i in range(len(initial_ious)):
                result_row[f'tool{i+1}_initial_iou'] = initial_ious[i]
                result_row[f'tool{i+1}_final_iou'] = final_ious[i]
            all_results.append(result_row)
            print(f"Successfully processed frame {frame_num}")
        except Exception as e:
            print(f"Error processing frame {frame_num}: {e}")
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_base_dir, 'iou_scores.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"IoU scores saved to: {csv_path}")
    
    print("Processing complete. Final point clouds saved to:", output_base_dir)