import numpy as np
import open3d as o3d
import cv2
from scipy.spatial import ConvexHull
import os
import glob
import pandas as pd


def perspective_project(points, K):
    """Project 3D points to 2D using perspective projection"""
    points_2d = K @ points.T
    points_2d = points_2d / points_2d[2]
    return points_2d[:2].T

def transform_points(points, T_source, T_target):
    """Transform points from source camera frame to target camera frame"""
    # Convert to homogeneous coordinates
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    # import ipdb; ipdb.set_trace()
    # Compute transformation matrix from source to target
    T_source_inv = np.linalg.inv(T_source)
    T_transform = T_target @ T_source_inv
    
    # Transform points
    points_transformed = (T_transform @ points_homo.T).T
    return points_transformed[:, :3]  # Return only x,y,z coordinates

def calculate_orthographic_matrix(points):
    """Calculate orthographic projection matrix based on point bounds"""
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

def calculate_scale_factor(bg_points, tool_points, ortho_matrix):
    """Calculate scale factor between background and tool point clouds"""
    # Project background points
    bg_homogeneous = np.concatenate([bg_points, np.ones((bg_points.shape[0], 1))], axis=1)
    bg_ortho = (ortho_matrix @ bg_homogeneous.T).T[:, :2]
    
    # Project tool points
    tool_homogeneous = np.concatenate([tool_points, np.ones((tool_points.shape[0], 1))], axis=1)
    tool_ortho = (ortho_matrix @ tool_homogeneous.T).T[:, :2]
    bg_area0 = (np.max(bg_ortho[:, 0]) - np.min(bg_ortho[:, 0])) * (np.max(bg_ortho[:, 1]) - np.min(bg_ortho[:, 1]))
    tool_area0 = (np.max(tool_ortho[:, 0]) - np.min(tool_ortho[:, 0])) * (np.max(tool_ortho[:, 1]) - np.min(tool_ortho[:, 1]))
    # Calculate areas using convex hull
    bg_area1 = calculate_precise_area(bg_ortho)
    tool_area1 = calculate_precise_area(tool_ortho)
    scale_factor0 = np.sqrt(bg_area0 / tool_area0)
    scale_factor1 = np.sqrt(bg_area1 / tool_area1)
    scale_factor = (scale_factor0 + scale_factor1) / 2

    # scale_factor = np.sqrt((bg_area0 + bg_area1) / (tool_area0 + tool_area1))/2
    
    return scale_factor, bg_ortho, tool_ortho

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
def create_masked_pointcloud(masked_depth, K):
    height, width = masked_depth.shape
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

def process_masked_region(mask, depth, K):
    """Process a masked region to create perspective point cloud"""
    y_coords, x_coords = np.where(mask == 255)
    pixels = np.stack([x_coords, y_coords, np.ones_like(x_coords)], axis=-1)
    depths = depth[y_coords, x_coords]
    
    # Convert to 3D points using perspective projection
    pixels_normalized = np.linalg.inv(K) @ pixels.T
    points_3d = pixels_normalized.T * depths[:, np.newaxis]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    return pcd, points_3d


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
    [569.4682, 0, 320],
    [0, 569.4682, 256],
    [0, 0, 1]
])

def get_pointcloud_pairs(pcd_dir):
    """Get all pairs of point clouds from directory"""
    # Define possible patterns for mask0 files
    patterns = [
        "*_mask0.ply",
        "*.color_mask0.ply"
    ]
    
    pairs = []
    
    # First try to find paired files
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
    
    # If no pairs found, look for individual numbered files
    if not pairs:
        numbered_files = sorted(glob.glob(os.path.join(pcd_dir, "[0-9]" * 6 + ".ply")))
        # Return single files instead of pairs
        for pcd_file in numbered_files:
            frame_num = os.path.basename(pcd_file).split(".")[0]
            pairs.append((frame_num, pcd_file, None))  # Set second file to None
    
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
            (bg_points[:, 0] - target_x)**2 + 
            (bg_points[:, 1] - target_y)**2
        )
        nearest_idx = np.argmin(distances)
        return int(bg_points[nearest_idx][2])


def process_sequence(pcd0_path, pcd1_path, depth_path, mask_path, output_path, frame_num):
    """Process a sequence of point clouds with transformations and scaling"""
    # Load point clouds
    pcd0 = o3d.io.read_point_cloud(pcd0_path)
    
    # Load mask and depth
    mask_rgb = cv2.imread(mask_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    # Find unique colors in mask
    unique_colors = np.unique(mask_rgb.reshape(-1, 3), axis=0)
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]
    
    # Process single or paired point clouds based on pcd1_path
    if pcd1_path is not None:
        pcd1 = o3d.io.read_point_cloud(pcd1_path)
        # Create masks for each tool
        mask0 = np.all(mask_rgb == unique_colors[0], axis=2).astype(np.uint8) * 255
        mask1 = np.all(mask_rgb == unique_colors[1], axis=2).astype(np.uint8) * 255
        tool_pairs = [(pcd0, mask0, 0), (pcd1, mask1, 1)]
    else:
        # Only process single point cloud
        mask0 = np.all(mask_rgb == unique_colors[0], axis=2).astype(np.uint8) * 255
        tool_pairs = [(pcd0, mask0, 0)]
    
    transformed_pcds = []
    results = []
    initial_iou_scores = []  # Add this line
    final_iou_scores = []    # Add this line
    initial_projections = [] # Add this line

    for pcd, mask, idx in tool_pairs:
        # 1. Transform points
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        transformed_points = transform_points(points, T_source, T_target)
        
        # Create transformed point cloud
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
        transformed_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 2. Get background points from depth
        _, bg_points = process_masked_region(mask, depth, K1)
        
        # 3. Calculate orthographic matrix and scale factor
        ortho_matrix = calculate_orthographic_matrix(bg_points)
        scale_factor, _, _ = calculate_scale_factor(bg_points, transformed_points, ortho_matrix)
        
        # 4. Scale the transformed points
        scaled_points = transformed_points * scale_factor

        # 5. Optimize position using perspective projection
        height, width = mask.shape
        tool_points = scaled_points  # Use scaled_points directly
        mask_binary = mask == 255
        gt_area = np.sum(mask_binary)
        
        # Optimize search ranges based on point cloud dimensions
        points_range = np.ptp(tool_points, axis=0)
        x_range = points_range[0]
        y_range = points_range[1]
        # Get maximum z value and its index
        # max_z_values = tool_points[:, 2]  # Get all z values
        # max_z_idx = np.argmax(max_z_values)  # Get index of maximum z
        # import ipdb; ipdb.set_trace()
        # Get corresponding x,y coordinates
        # max_point = tool_points[max_z_idx]
        # x_coord, y_coord, z_coord_tools = max_point
        # match_idx = get_z_at_xy(bg_points, x_coord, y_coord)
        # z_coord_bg = bg_points[match_idx][-1]

        # bg_points_mean = np.mean(bg_points, axis=0)
        # tool_points_mean = np.mean(tool_points, axis=0)
        # z_range = z_coord_bg - z_coord_tools
        # print('z_range',z_range)
        
        x_translations = np.linspace(-x_range/2, x_range/2, 10)
        y_translations = np.linspace(-y_range/2, y_range/2, 10)
        z_translations = np.linspace(-5, 15, 20)  # Keep z search range reasonable
        
        best_score = float('inf')
        best_transform = None
        best_projection = None
        best_iou = 0  # Add this line

        # Calculate initial IoU before optimization
        height, width = mask.shape
        tool_points = scaled_points
        mask_binary = mask == 255
        
        # Project initial points
        initial_projected_points = perspective_project(tool_points, K1)
        initial_proj_mask = np.zeros((height, width), dtype=bool)
        valid_points = (initial_projected_points[:, 0] >= 0) & (initial_projected_points[:, 0] < width) & \
                      (initial_projected_points[:, 1] >= 0) & (initial_projected_points[:, 1] < height)
        
        if np.any(valid_points):
            coords = initial_projected_points[valid_points].astype(int)
            initial_proj_mask[coords[:, 1], coords[:, 0]] = True
            initial_proj_mask = cv2.dilate(initial_proj_mask.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
            
            initial_intersection = np.sum(initial_proj_mask & mask_binary)
            initial_union = np.sum(initial_proj_mask | mask_binary)
            initial_iou = initial_intersection / initial_union if initial_union > 0 else 0
            initial_iou_scores.append(initial_iou)
            initial_projections.append(initial_proj_mask)
        else:
            initial_iou_scores.append(0.0)
            initial_projections.append(np.zeros((height, width), dtype=bool))
        
        # Vectorized position optimization
        for x in x_translations:
            for y in y_translations:
                for z in z_translations:
                    # Apply translation
                    translated_points = tool_points + np.array([x, y, z])
                    
                    # Project points efficiently
                    projected_points = perspective_project(translated_points, K1)
                    
                    # Create and evaluate projection mask
                    proj_mask = np.zeros((height, width), dtype=bool)
                    valid_points = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < width) & \
                                 (projected_points[:, 1] >= 0) & (projected_points[:, 1] < height)
                    
                    if np.any(valid_points):
                        coords = projected_points[valid_points].astype(int)
                        proj_mask[coords[:, 1], coords[:, 0]] = True
                        
                        # Quick dilation for better coverage
                        proj_mask = cv2.dilate(proj_mask.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool)
                        
                        # Compute IoU instead of just area difference
                        intersection = np.sum(proj_mask & mask_binary)
                        union = np.sum(proj_mask | mask_binary)
                        score = 1.0 - (intersection / union if union > 0 else 0)
                        
                        if score < best_score:
                            best_score = score
                            best_transform = np.array([x, y, z])
                            best_projection = proj_mask
                            best_iou = intersection / union if union > 0 else 0  # Store IoU directly
        
        # After optimization, store final IoU
        final_iou_scores.append(best_iou)
        
        print(f"Tool {idx} - Initial IoU: {initial_iou_scores[-1]:.4f}")
        print(f"Tool {idx} - Final IoU: {best_iou:.4f}")
        
        # Apply best transformation
        final_points = tool_points + best_transform
        transformed_pcd = o3d.geometry.PointCloud()
        transformed_pcd.points = o3d.utility.Vector3dVector(final_points)
        transformed_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        
        transformed_pcds.append(transformed_pcd)
        results.append((best_transform[0], best_transform[1], best_transform[2], best_projection))
        
        # Save individual transformed point cloud
        output_path = os.path.join(output_base_dir, f"final_tool{frame_num}.ply")
        o3d.io.write_point_cloud(output_path, transformed_pcd)
        print(f"Tool {idx} - Scale factor: {scale_factor:.4f}")
        print(f"Tool {idx} - Best position: X={best_transform[0]:.4f}, Y={best_transform[1]:.4f}, Z={best_transform[2]:.4f}")
        print(f"Tool {idx} - Best IoU score: {1-best_score:.4f}")
    
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
        combined_pcd = transformed_pcds[0] + transformed_pcds[1]
    else:
        # Single point cloud case
        final_vis = np.zeros((height, width, 3), dtype=np.uint8)
        final_vis[mask0 == 255] = [255, 0, 0]
        final_vis[results[0][3]] = [255, 0, 0]
        final_vis[(mask0 == 255) & results[0][3]] = [255, 255, 0]
        cv2.imwrite(os.path.join(output_base_dir, f"projection_comparison_final{frame_num}.png"), final_vis)
        combined_pcd = transformed_pcds[0]

    o3d.io.write_point_cloud(output_path, combined_pcd)

    # After processing tools, save initial projection visualization
    if len(initial_projections) >= 2:
        initial_vis = np.zeros((height, width, 3), dtype=np.uint8)
        initial_vis[mask0 == 255] = [255, 0, 0]      # Ground truth tool1 in blue
        initial_vis[mask1 == 255] = [255, 0, 0]      # Ground truth tool2 in blue
        initial_vis[initial_projections[0]] = [0, 255, 0]    # Initial projection tool1 in green
        initial_vis[initial_projections[1]] = [0, 255, 0]  # Initial projection tool2 in green
        initial_vis[(mask0 == 255) & initial_projections[0]] = [255, 255, 0]   # Overlap tool1 in yellow
        initial_vis[(mask1 == 255) & initial_projections[1]] = [255, 255, 0] # Overlap tool2 in white
        cv2.imwrite(os.path.join(output_base_dir, f"projection_comparison_initial{frame_num}.png"), initial_vis)
    elif len(initial_projections) == 1:
        initial_vis = np.zeros((height, width, 3), dtype=np.uint8)
        initial_vis[mask0 == 255] = [255, 0, 0]
        initial_vis[initial_projections[0]] = [0, 255, 0]
        initial_vis[(mask0 == 255) & initial_projections[0]] = [255, 255, 0]
        cv2.imwrite(os.path.join(output_base_dir, f"projection_comparison_initial{frame_num}.png"), initial_vis)
    
    return transformed_pcds, initial_iou_scores, final_iou_scores

if __name__ == "__main__":
    # Define base directories
    pcd_base_dir = "/workspace/EndoLRM2/endonerf/cutting/zxhezexin/openlrm-mix-base-1.1/meshes"
    depth_base_dir = "/workspace/EndoLRM2/endonerf/cutting/zxhezexin/openlrm-mix-base-1.1/rendered_depth"
    mask_base_dir = "/workspace/dataset/endolrm_dataset/endonerf/cutting_tissues_twice/Annotations"
    output_base_dir = "/workspace/EndoLRM2/endonerf/cutting/zxhezexin/openlrm-mix-base-1.1/final_tools"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    all_results = []
    # Get all pairs of point clouds
    pointcloud_pairs = get_pointcloud_pairs(pcd_base_dir)
    # import ipdb; ipdb.set_trace()
    # Process each pair
    for frame_num, pcd0_path, pcd1_path in pointcloud_pairs:
        # import ipdb; ipdb.set_trace()
        depth_path = os.path.join(depth_base_dir, f"{frame_num}_endo_depth.png")
        mask_path = os.path.join(mask_base_dir, f"{frame_num}.png")
        output_path = os.path.join(output_base_dir, f"frame_{frame_num}_combined.ply")
        
        if os.path.exists(depth_path) and os.path.exists(mask_path):
            print(f"Processing frame {frame_num}...")
            try:
                final_pcds, initial_ious, final_ious = process_sequence(pcd0_path, pcd1_path, depth_path, mask_path, output_path, frame_num)
                result_dict = {'frame': frame_num}
                if len(initial_ious) >= 2:
                    result_dict.update({
                        'tool1_initial_iou': initial_ious[0],
                        'tool2_initial_iou': initial_ious[1],
                        'tool1_final_iou': final_ious[0],
                        'tool2_final_iou': final_ious[1]
                    })
                else:
                    result_dict.update({
                        'tool1_initial_iou': initial_ious[0],
                        'tool1_final_iou': final_ious[0]
                    })
                all_results.append(result_dict)
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

