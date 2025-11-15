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
    """Get all pairs of point clouds from directory"""
    # Define possible patterns for mask0 files
    patterns = [
        "*_mask0.ply",
        "*.color_mask0.ply"
    ]
    
    pairs = []
    for pattern in patterns:
        pcd0_files = sorted(glob.glob(os.path.join(pcd_dir, pattern)))
        
        for pcd0_file in pcd0_files:
            # Handle different naming formats
            if "stereomis" in pcd0_file:
                pcd1_file = pcd0_file.replace("_mask0.ply", "_mask1.ply")
                frame_num = os.path.basename(pcd0_file).split("_")[1]
            elif "endonerf" in pcd0_file:
                pcd1_file = pcd0_file.replace(".color_mask0.ply", ".color_mask1.ply")
                frame_num = os.path.basename(pcd0_file).split("-")[1].split(".")[0]
            
            if os.path.exists(pcd1_file):
                pairs.append((frame_num, pcd0_file, pcd1_file))
    
    return sorted(pairs)


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


def process_sequence(pcd0_path, pcd1_path, depth_path, mask_path, output_path, frame_num):
    """Process a sequence of point clouds with transformations and scaling"""
    # Load point clouds
    pcd0 = o3d.io.read_point_cloud(pcd0_path)
    pcd1 = o3d.io.read_point_cloud(pcd1_path)

    # Load mask and depth
    mask_rgb = cv2.imread(mask_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    import ipdb; ipdb.set_trace()
    # Find unique colors in mask
    unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]
    
    # Create masks for each tool
    mask0 = np.all(mask_rgb == unique_colors[0], axis=2).astype(np.uint8) * 255
    mask1 = np.all(mask_rgb == unique_colors[1], axis=2).astype(np.uint8) * 255
    
    # Process each tool
    tool_pairs = [
        (pcd0, mask0, 0),
        (pcd1, mask1, 1)
    ]
    
    transformed_pcds = []
    results = []
    initial_iou_scores = []
    final_iou_scores = []
    initial_projections = []
    initial_transformed_pcds = []
    
    for pcd, mask, idx in tool_pairs:
        # 1. Transform points
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        transformed_points = transform_points(points, T_source, T_target, use_cuda=CUDA_AVAILABLE)
        
        # Create transformed point cloud
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
        transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Store initial transformed point cloud for later combination
        initial_transformed_pcds.append(transformed_pcd)
        
        # 2. Get background points from depth
        _, bg_points = process_masked_region(mask, depth, K1, use_cuda=CUDA_AVAILABLE)
        
        # 3. Calculate orthographic matrix and scale factor
        ortho_matrix = calculate_orthographic_matrix(bg_points, use_cuda=CUDA_AVAILABLE)
        scale_factor, _, _ = calculate_scale_factor(bg_points, transformed_points, ortho_matrix, use_cuda=CUDA_AVAILABLE)
        
        # 4. Scale the transformed points
        scaled_points = transformed_points * scale_factor

        # Calculate initial IoU before optimization
        height, width = mask.shape
        tool_points = scaled_points
        mask_binary = mask == 255
        
        # Project initial points
        initial_projected_points = perspective_project(tool_points, K1, use_cuda=CUDA_AVAILABLE)
        initial_proj_mask = np.zeros((height, width), dtype=bool)
        valid_points = (initial_projected_points[:, 0] >= 0) & (initial_projected_points[:, 0] < width) & \
                      (initial_projected_points[:, 1] >= 0) & (initial_projected_points[:, 1] < height)
        
        if np.any(valid_points):
            coords = initial_projected_points[valid_points].astype(int)
            initial_proj_mask[coords[:, 1], coords[:, 0]] = True
            initial_proj_mask = cv2.dilate(initial_proj_mask.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
            
            # Calculate initial IoU
            initial_intersection = np.sum(initial_proj_mask & mask_binary)
            initial_union = np.sum(initial_proj_mask | mask_binary)
            initial_iou = initial_intersection / initial_union if initial_union > 0 else 0
            initial_iou_scores.append(initial_iou)
            initial_projections.append(initial_proj_mask)
        else:
            initial_iou_scores.append(0.0)
            initial_projections.append(np.zeros((height, width), dtype=bool))
            
        # Optimize position using perspective projection with CUDA acceleration
        height, width = mask.shape
        tool_points = scaled_points
        mask_binary = mask == 255
        gt_area = np.sum(mask_binary)
        
        # Optimize search ranges based on point cloud dimensions
        if CUDA_AVAILABLE:
            tool_points_gpu = cp.asarray(tool_points)
            points_range = cp.asnumpy(cp.ptp(tool_points_gpu, axis=0))
            max_z_idx = int(cp.argmax(tool_points_gpu[:, 2]))
        else:
            points_range = np.ptp(tool_points, axis=0)
            max_z_idx = np.argmax(tool_points[:, 2])
            
        x_range = points_range[0]
        y_range = points_range[1]
        
        # Get maximum z value and its index
        max_point = tool_points[max_z_idx]
        x_coord, y_coord, z_coord_tools = max_point
        match_idx = get_z_at_xy(bg_points, x_coord, y_coord)
        z_coord_bg = bg_points[match_idx][-1]
        z_range = z_coord_bg - z_coord_tools
        
        x_translations = np.linspace(-x_range/2, x_range/2, 10)
        y_translations = np.linspace(-y_range/2, y_range/2, 10)
        z_translations = np.linspace(0, z_range, 20)
        
        best_score = float('inf')
        best_transform = None
        best_projection = None
        best_iou = 0
        
        # GPU-accelerated position optimization
        if CUDA_AVAILABLE:
            tool_points_gpu = cp.asarray(tool_points)
            K1_gpu = cp.asarray(K1)
            mask_binary_gpu = cp.asarray(mask_binary)
            
            for x in x_translations:
                for y in y_translations:
                    for z in z_translations:
                        # Apply translation on GPU
                        translated_points = tool_points_gpu + cp.array([x, y, z])
                        
                        # Project points efficiently on GPU
                        points_2d = K1_gpu @ translated_points.T
                        points_2d = points_2d / points_2d[2]
                        projected_points = cp.asnumpy(points_2d[:2].T)
                        
                        # Create and evaluate projection mask
                        proj_mask = np.zeros((height, width), dtype=bool)
                        valid_points = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) & \
                                     (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)
                        
                        if np.any(valid_points):
                            coords = projected_points[valid_points].astype(int)
                            proj_mask[coords[:, 1], coords[:, 0]] = True
                            proj_mask = cv2.dilate(proj_mask.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
                            
                            # Compute IoU
                            intersection = np.sum(proj_mask & mask_binary)
                            union = np.sum(proj_mask | mask_binary)
                            iou = intersection / union if union > 0 else 0
                            score = 1.0 - iou
                            
                            if score < best_score:
                                best_score = score
                                best_transform = np.array([x, y, z])
                                best_projection = proj_mask
                                best_iou = iou
        else:
            # CPU fallback
            for x in x_translations:
                for y in y_translations:
                    for z in z_translations:
                        # Apply translation
                        translated_points = tool_points + np.array([x, y, z])
                        
                        # Project points efficiently
                        projected_points = perspective_project(translated_points, K1, use_cuda=False)
                        
                        # Create and evaluate projection mask
                        proj_mask = np.zeros((height, width), dtype=bool)
                        valid_points = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) & \
                                     (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)
                        
                        if np.any(valid_points):
                            coords = projected_points[valid_points].astype(int)
                            proj_mask[coords[:, 1], coords[:, 0]] = True
                            proj_mask = cv2.dilate(proj_mask.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
                            
                            # Compute IoU
                            intersection = np.sum(proj_mask & mask_binary)
                            union = np.sum(proj_mask | mask_binary)
                            iou = intersection / union if union > 0 else 0
                            score = 1.0 - iou
                            
                            if score < best_score:
                                best_score = score
                                best_transform = np.array([x, y, z])
                                best_projection = proj_mask
                                best_iou = iou
        
        # Apply best transformation
        final_points = tool_points + best_transform
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(final_points)
        transformed_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        
        transformed_pcds.append(transformed_pcd)
        results.append((best_transform[0], best_transform[1], best_transform[2], best_projection))
        
        final_iou_scores.append(best_iou)
        
        print(f"Tool {idx} - Initial IoU: {initial_iou_scores[-1]:.4f}")
        print(f"Tool {idx} - Final IoU: {best_iou:.4f}")
    
    # Save individual and combined initial transformed point clouds
    initial_tools_dir = os.path.join(output_base_dir.replace('postprocessed_tools', 'initial_tools'))
    os.makedirs(initial_tools_dir, exist_ok=True)
    
    if len(initial_transformed_pcds) >= 2:
        # # Save individual initial transformed point clouds
        # for idx, initial_pcd in enumerate(initial_transformed_pcds):
        #     initial_output_path = os.path.join(initial_tools_dir, f"frame_{frame_num}_tool{idx}_initial.ply")
        #     o3d.io.write_point_cloud(initial_output_path, initial_pcd)
        
        # Combine and save initial transformed point clouds
        combined_initial_pcd = initial_transformed_pcds[0] + initial_transformed_pcds[1]
        initial_combined_path = os.path.join(initial_tools_dir, f"frame_{frame_num}_combined_initial.ply")
        o3d.io.write_point_cloud(initial_combined_path, combined_initial_pcd)
    
    # Create visualization
    if len(results) >= 2:
        final_vis = np.zeros((height, width, 3), dtype=np.uint8)
        final_vis[mask0 == 255] = [255, 0, 0]
        final_vis[mask1 == 255] = [255, 0, 0]
        final_vis[results[0][3]] = [0, 255, 0]
        final_vis[results[1][3]] = [0, 255, 0]
        final_vis[(mask0 == 255) & results[0][3]] = [255, 255, 0]
        final_vis[(mask1 == 255) & results[1][3]] = [255, 255, 0]
        
        cv2.imwrite(os.path.join(output_base_dir, f"projection_comparison_final{frame_num}.png"), final_vis)
    
    # After processing both tools, create and save initial projection visualization
    if len(initial_projections) >= 2:
        initial_vis = np.zeros((height, width, 3), dtype=np.uint8)
        initial_vis[mask0 == 255] = [255, 0, 0]      # Ground truth tool1 in red
        initial_vis[mask1 == 255] = [255, 0, 0]      # Ground truth tool2 in blue
        initial_vis[initial_projections[0]] = [0, 255, 0]    # Initial projection tool1 in green
        initial_vis[initial_projections[1]] = [0, 255, 0]  # Initial projection tool2 in cyan
        initial_vis[(mask0 == 255) & initial_projections[0]] = [255, 255, 0]   # Overlap tool1 in yellow
        initial_vis[(mask1 == 255) & initial_projections[1]] = [255, 255, 0] # Overlap tool2 in white
        
        cv2.imwrite(os.path.join(output_base_dir, f"projection_comparison_initial{frame_num}.png"), initial_vis)
    
    # Instead of saving individual clouds, combine them
    combined_pcd = transformed_pcds[0] + transformed_pcds[1]
    o3d.io.write_point_cloud(output_path, combined_pcd)
    
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

K1 = np.array([
    [1033.894287109375, 0.0, 604.578857421875],
    [0.0, 1033.7147216796875, 514.9761962890625],
    [0.0, 0.0, 1.0]
])


if __name__ == "__main__":
    # # Define base directories
    pcd_base_dir = "/workspace/EndoLRMGS/ablation_study/stereomis/v1/zxhezexin/openlrm-mix-base-1.1/meshes"
    depth_base_dir = "/workspace/EndoLRMGS/ablation_study/stereomis/v1/zxhezexin/openlrm-mix-base-1.1/rendered_depth"
    mask_base_dir = "/workspace/datasets/endolrm_dataset/stereomis/p2_6/Annotations_v5"
    output_base_dir = "/workspace/EndoLRMGS/ablation_study/stereomis/v1/zxhezexin/postprocessed_tools"

    # Create output folder if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all pairs of point clouds
    pointcloud_pairs = get_pointcloud_pairs(pcd_base_dir)
    
    # Create lists to store results
    all_results = []
    
    # Process each pair
    for frame_num, pcd0_path, pcd1_path in pointcloud_pairs:
        # import ipdb; ipdb.set_trace()
        depth_path = os.path.join(depth_base_dir, f"frame_{frame_num}_endo_depth.png")
        mask_path = os.path.join(mask_base_dir, f"frame_{frame_num}.png")
        output_path = os.path.join(output_base_dir, f"frame_{frame_num}_combined.ply")
        
        if os.path.exists(depth_path) and os.path.exists(mask_path):
            print(f"Processing frame {frame_num}...")
            try:
                final_pcds, initial_ious, final_ious = process_sequence(pcd0_path, pcd1_path, depth_path, mask_path, output_path, frame_num)
                all_results.append({
                    'frame': frame_num,
                    'tool1_initial_iou': initial_ious[0],
                    'tool2_initial_iou': initial_ious[1],
                    'tool1_final_iou': final_ious[0],
                    'tool2_final_iou': final_ious[1]
                })
                print(f"Successfully processed frame {frame_num}")
            except Exception as e:
                print(f"Error processing frame {frame_num}: {e}")
        else:
            print(f"Missing depth or mask file for frame {frame_num}")
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_base_dir, 'iou_scores.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"IoU scores saved to: {csv_path}")
    
    print("Processing complete. Final point clouds saved to:", output_base_dir)