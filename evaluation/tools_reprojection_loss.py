import numpy as np
import cv2
import json
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import json
import pandas as pd
from PIL import Image


def load_calibration_frame_json(frame_json_path, image_size_hw):
    """Load original stereo calibration and compute rectified right intrinsics/extrinsics (alpha = -1).
    
    Returns:
        KL_orig (3x3), KL_rect (3x3), P2 (3x4), R_left_rect (3x3), R_right (3x3), T_right (3x1)
    """
    with open(frame_json_path, 'r') as f:
        data = json.load(f)

    calib = data.get("camera-calibration", data.get("camera_calibration", data))

    def get_first_available(d, keys, default=None):
        for k in keys:
            if k in d:
                return d[k]
        return default

    # Original intrinsics
    KL = np.array(get_first_available(calib, ["KL", "K_left", "K1", "K"]), dtype=np.float32)
    KR = np.array(get_first_available(calib, ["KR", "K_right", "K2"]), dtype=np.float32)

    # Distortions
    DL = get_first_available(calib, ["DL", "D_left", "D1", "D"])
    DR = get_first_available(calib, ["DR", "D_right", "D2"])
    if DL is None:
        DL = np.zeros((5, 1), dtype=np.float32)
    else:
        DL = np.array(DL, dtype=np.float32).reshape(-1, 1)
    if DR is None:
        DR = np.zeros((5, 1), dtype=np.float32)
    else:
        DR = np.array(DR, dtype=np.float32).reshape(-1, 1)

    # Stereo extrinsics
    R = np.array(get_first_available(calib, ["R", "R_lr", "R_left_to_right"]), dtype=np.float32)
    T = np.array(get_first_available(calib, ["T", "T_lr", "T_left_to_right"]), dtype=np.float32).reshape(3, 1)

    h, w = image_size_hw
    image_size_wh = (w, h)

    # Compute rectification (alpha = -1)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        KL, DL, KR, DR, image_size_wh, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1
    )
    
    
    KL_orig = KL.copy().astype(np.float32)
    KL_rect = P1[:3, :3].astype(np.float32)
    
    # 返回完整的 P2，包含平移分量
    R_left_rect = R1.astype(np.float32)
    R_right = R2.astype(np.float32)
    T_right = T.astype(np.float32)
    
    return KL_orig, KL_rect, P2.astype(np.float32), R_left_rect, R_right, T_right

def calculate_iou(projection_mask, gt_mask):
    """Calculate IoU between projection mask and ground truth mask
    
    Args:
        projection_mask: Binary mask (H, W), bool or uint8
        gt_mask: Binary mask (H, W), bool or uint8
    
    Returns:
        iou: float, IoU score
    """
    proj_binary = projection_mask.astype(bool) if projection_mask.dtype != bool else projection_mask
    gt_binary = gt_mask.astype(bool) if gt_mask.dtype != bool else gt_mask
    
    intersection = np.sum(proj_binary & gt_binary)
    union = np.sum(proj_binary | gt_binary)
    
    iou = intersection / union if union > 0 else 0.0
    return iou

def create_color_visualization(projection_mask, gt_mask, output_path):
    """Create color-coded visualization of projection vs ground truth
    
    Args:
        projection_mask: Binary mask (H, W), bool or uint8
        gt_mask: Binary mask (H, W), bool or uint8  
        output_path: Path to save visualization
    
    Color coding:
        Blue: Ground truth only
        Green: Projection only
        Cyan: Overlap (both GT and projection)
    """
    h, w = gt_mask.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    proj_binary = projection_mask.astype(bool)
    gt_binary = gt_mask.astype(bool)
    
    # Blue: GT only
    vis[gt_binary] = [0, 0, 255]
    # Green: Projection only  
    vis[proj_binary] = [0, 255, 0]
    # Cyan: Overlap
    vis[gt_binary & proj_binary] = [0, 255, 255]
    
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

def project_pointcloud_generic(point_cloud_path, K_or_P, image_size, R_rel=None, T_rel=None, output_path=None, interpolate=False, flip_z=False, use_projection_matrix=False, inpaint_near_points=False, near_dist_thresh=6, gt_mask=None):
    """Project point cloud into target view.
    
    Args:
        use_projection_matrix: If True, K_or_P is a 3x4 projection matrix; otherwise 3x3 intrinsic matrix
        gt_mask: Optional ground truth mask (H, W), uint8 where 255 = valid region. If provided, masked areas are set to white before projection.
        
    Returns:
        img: Rendered image
        projection_mask: Binary mask of projected pixels (before inpainting)
    """
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    if flip_z:
        flip_matrix = np.diag([1, 1, -1]).astype(np.float32)
        points = (flip_matrix @ points.T).T

    if use_projection_matrix:
        # K_or_P is P2 (3x4), apply rotation first, then use P2 directly
        if R_rel is not None and T_rel is not None:
            pts = (R_rel @ points.T + T_rel).T
        else:
            pts = points
        
        # Add homogeneous coordinate
        pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
        proj = (K_or_P @ pts_homo.T)  # 3 x N
        proj /= proj[2:3, :]
        uv = proj[:2, :].T
        depths = pts[:, 2]
    else:
        # Original behavior
        if R_rel is not None and T_rel is not None:
            pts = (R_rel @ points.T + T_rel).T
        else:
            pts = points

        depths = pts[:, 2]
        proj = (K_or_P @ pts.T)
        proj /= proj[2:3, :]
        uv = proj[:2, :].T

    depth_mask = depths > 0
    uv = uv[depth_mask]
    colors = colors[depth_mask]
    depths = depths[depth_mask]

    h, w = image_size
    in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[in_bounds]
    colors = colors[in_bounds]
    depths = depths[in_bounds]

    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Set ground truth masked areas to white if mask is provided
    if gt_mask is not None:
        white_mask = gt_mask > 0
        img[white_mask] = 255

    depth_buffer = np.full((h, w), np.inf, dtype=np.float32)

    px = uv.astype(int)
    cols = (colors * 255).astype(np.uint8)

    # Create projection mask before inpainting
    projection_mask = np.zeros((h, w), dtype=bool)
    
    for (x, y), c, d in zip(px, cols, depths):
        if d < depth_buffer[y, x]:
            img[y, x] = c
            depth_buffer[y, x] = d
            projection_mask[y, x] = True

    # Dilate projection mask slightly to match postprocess approach
    projection_mask = cv2.dilate(projection_mask.astype(np.uint8), np.ones((5,5), np.uint8)).astype(bool)

    if interpolate:
        # Hole = no projection (depth_buffer == inf)
        hole_mask = (depth_buffer == np.inf)
        if hole_mask.any():
            if inpaint_near_points:
                # Distance (in pixels) from each hole to nearest valid pixel
                hole_uint8 = hole_mask.astype(np.uint8)
                dist_map = cv2.distanceTransform(hole_uint8, cv2.DIST_L2, 3)
                local_holes = (hole_mask & (dist_map <= near_dist_thresh))
                inpaint_mask = np.zeros((h, w), dtype=np.uint8)
                inpaint_mask[local_holes] = 255
            else:
                inpaint_mask = hole_mask.astype(np.uint8) * 255

            if inpaint_mask.sum() > 0:
                img = cv2.inpaint(img, inpaint_mask, 3, cv2.INPAINT_TELEA)

    if output_path:
        # cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        Image.fromarray(img).save(output_path)
    
    return img, projection_mask

def calculate_masked_metrics(image1, image2, mask, lpips_model=lpips.LPIPS(net='alex').cuda()):
    # Apply mask to both images
    masked_image1 = image1.astype(np.float32)
    masked_image2 = image2.astype(np.float32)
    # Set unmasked areas to 0
    mask = (mask / 255.0).astype(np.float32)  # Ensure mask is also float32
    masked_image1 = masked_image1 #* mask[:, :, np.newaxis]
    masked_image2 = masked_image2 * mask[:, :, np.newaxis]
    
    # Convert back to uint8 for OpenCV operations
    masked_image1_uint8 = masked_image1.astype(np.uint8)
    masked_image2_uint8 = masked_image2.astype(np.uint8)
    
    cv2.imwrite('masked_image1.png', masked_image1_uint8)
    cv2.imwrite('masked_image2.png', masked_image2_uint8)
    
    # Calculate metrics only on unmasked areas
    psnr_value = psnr(masked_image1, masked_image2, data_range=255)
    
    gray1 = cv2.cvtColor(masked_image1_uint8, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(masked_image2_uint8, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(gray1, gray2)
    
    # For LPIPS, we need to handle the mask differently
    image1_tensor = lpips.im2tensor(masked_image1_uint8).cuda()
    image2_tensor = lpips.im2tensor(masked_image2_uint8).cuda()
    lpips_value = lpips_model(image1_tensor, image2_tensor).item()
    
    return psnr_value, ssim_value, lpips_value

# def calculate_masked_metrics(image1, image2, mask, lpips_model=lpips.LPIPS(net='alex').cuda()):
#     """Calculate PSNR, SSIM, and LPIPS only within masked regions.
    
#     Args:
#         image1: RGB image (H, W, 3), uint8 or float32
#         image2: RGB image (H, W, 3), uint8 or float32
#         mask: Binary mask (H, W), uint8 where 255 = valid region
#         lpips_model: Pre-initialized LPIPS model
    
#     Returns:
#         psnr_value, ssim_value, lpips_value (all float or None if no valid pixels)
#     """

#     # Create binary mask
#     valid_pixels = mask > 0
#     if not valid_pixels.any():
#         return None, None, None
    
#     # Convert images to float32 for calculations
#     img1_float = image1.astype(np.float32)
#     img2_float = image2.astype(np.float32)
    
#     # --- PSNR Calculation (only on masked pixels) ---
#     # Extract only valid pixels
#     valid_img1 = img1_float[valid_pixels]
#     valid_img2 = img2_float[valid_pixels]
    
#     # Calculate MSE only on valid pixels
#     mse = np.mean((valid_img1 - valid_img2) ** 2)
#     if mse == 0:
#         psnr_value = float('inf')
#     else:
#         psnr_value = 20 * np.log10(255.0 / np.sqrt(mse))
    
#     # --- SSIM Calculation (only on masked pixels) ---
#     # Convert to grayscale
#     gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY).astype(np.float32)
#     gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
#     # Calculate SSIM with full map to get per-pixel scores
#     ssim_map = ssim(gray1, gray2, data_range=255, full=True)[1]
    
#     # Average SSIM only over masked pixels
#     ssim_value = np.mean(ssim_map[valid_pixels])
    
#     # --- LPIPS Calculation ---
#     # Crop to bounding box of mask to reduce computation
#     rows = np.any(valid_pixels, axis=1)
#     cols = np.any(valid_pixels, axis=0)
#     ymin, ymax = np.where(rows)[0][[0, -1]]
#     xmin, xmax = np.where(cols)[0][[0, -1]]
    
#     # Crop images and mask
#     crop_img1 = image1[ymin:ymax+1, xmin:xmax+1].astype(np.uint8)
#     crop_img2 = image2[ymin:ymax+1, xmin:xmax+1].astype(np.uint8)
#     crop_mask = valid_pixels[ymin:ymax+1, xmin:xmax+1]
    
#     # Create masked versions for LPIPS (set invalid regions to black)
#     masked_crop1 = crop_img1.copy()
#     masked_crop2 = crop_img2.copy()
#     masked_crop1[~crop_mask] = 0
#     masked_crop2[~crop_mask] = 0
    
#     # Convert to tensors and compute LPIPS
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     image1_tensor = lpips.im2tensor(masked_crop1).to(device)
#     image2_tensor = lpips.im2tensor(masked_crop2).to(device)
    
#     with torch.no_grad():
#         lpips_value = lpips_model(image1_tensor, image2_tensor).item()
    
#     return psnr_value, ssim_value, lpips_value


def main():
    # Paths
    pointcloud_dir = Path('/workspace/EndoLRMGS/stereomis/p1/initial_tools')
    frame_data_file = Path('/workspace/datasets/endolrm_dataset/stereomis/p1/frame_data.json')

    left_mask_dir = Path('/workspace/datasets/endolrm_dataset/stereomis/p1/binary_mask_deva')
    left_gt_image_dir = Path('/workspace/datasets/endolrm_dataset/stereomis/p1/left_finalpass')
    right_mask_dir = Path('/workspace/Tracking-Anything-with-DEVA/output/stereomis/p1/right_view/binary_mask_deva')
    right_gt_image_dir = Path('/workspace/datasets/endolrm_dataset/stereomis/p1/right_finalpass')

    left_output_dir = Path('/workspace/EndoLRMGS/stereomis/p1/init_left_view_reprojected_images')
    right_output_dir = Path('/workspace/EndoLRMGS/stereomis/p1/init_right_view_reprojected_images')
    left_vis_dir = Path('/workspace/EndoLRMGS/stereomis/p1/init_left_view_iou_visualizations')
    right_vis_dir = Path('/workspace/EndoLRMGS/stereomis/p1/init_right_view_iou_visualizations')
    
    left_output_dir.mkdir(exist_ok=True)
    right_output_dir.mkdir(exist_ok=True)
    left_vis_dir.mkdir(exist_ok=True)
    right_vis_dir.mkdir(exist_ok=True)

    # Infer image size from left GT (rectified images)
    left_gts = sorted(left_gt_image_dir.glob('*.png'))
    if len(left_gts) == 0:
        print("No left GT images found.")
        return
    sample_left = cv2.imread(str(left_gts[0]))
    h, w = sample_left.shape[:2]
    image_size = (h, w)

    # Compute rectified calibration from original parameters (alpha = -1)
    KL_orig, KL_rect, P2, R_left_rect, R_right, T_right = load_calibration_frame_json(frame_data_file, image_size)

    print("Calibration:")
    print(f"KL_orig (left, retained):\n{KL_orig}")
    print(f"KL_rect:\n{KL_rect}")
    print(f"R_left_rect:\n{R_left_rect}")
    print("Right rectified (with translation):")
    print(f"P2 (complete projection matrix):\n{P2}")
    print(f"R_right:\n{R_right}")
    print(f"T_right.T:\n{T_right.T}")

    # Prepare lists
    right_gts = sorted(right_gt_image_dir.glob('*.png'))
    left_masks = sorted(left_mask_dir.glob('*.png'))
    right_masks = sorted(right_mask_dir.glob('*.png'))
    point_clouds = sorted(pointcloud_dir.glob('*.ply'))

    min_count = min(len(point_clouds), len(left_gts), len(right_gts), len(left_masks), len(right_masks))
    if min_count == 0:
        print("Insufficient data to proceed.")
        return
    if any(len(lst) != min_count for lst in [point_clouds, left_gts, right_gts, left_masks, right_masks]):
        print(f"Count mismatch detected. Using first {min_count} items.")

    # LPIPS model
    lpips_model = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')
    lpips_model.eval()

    # Store results
    all_results = []

    for idx in tqdm(range(min_count)):
        pcd_path = point_clouds[idx]
        frame_name = pcd_path.stem

        # Load GT masks first
        left_mask_path = left_masks[idx]
        right_mask_path = right_masks[idx]
        left_gt_mask_original = cv2.imread(str(left_mask_path), cv2.IMREAD_GRAYSCALE)
        right_gt_mask_original = cv2.imread(str(right_mask_path), cv2.IMREAD_GRAYSCALE)

        # Left projection: use mask for white background
        left_img, left_proj_mask = project_pointcloud_generic(
            pcd_path, KL_orig, image_size, None, None,
            str(left_output_dir / f"{frame_name}_left.png"), interpolate=True, flip_z=False,
            inpaint_near_points=True, near_dist_thresh=5, gt_mask=left_gt_mask_original
        )

        # Right projection: use mask for white background
        right_img, right_proj_mask = project_pointcloud_generic(
            pcd_path, P2, image_size, None, None,
            str(right_output_dir / f"{frame_name}_right.png"), 
            interpolate=True, use_projection_matrix=True, flip_z=False,
            inpaint_near_points=True, near_dist_thresh=5, gt_mask=right_gt_mask_original
        )
        
        # Load GT images
        left_gt_img_path = left_gts[idx]
        right_gt_img_path = right_gts[idx]
        
        left_gt_img = cv2.imread(str(left_gt_img_path))
        right_gt_img = cv2.imread(str(right_gt_img_path))
        
        # Convert BGR to RGB for consistency
        left_gt_img = cv2.cvtColor(left_gt_img, cv2.COLOR_BGR2RGB)
        right_gt_img = cv2.cvtColor(right_gt_img, cv2.COLOR_BGR2RGB)
        
        # Calculate IoU scores using ORIGINAL masks
        left_iou = calculate_iou(left_proj_mask, left_gt_mask_original == 255)
        right_iou = calculate_iou(right_proj_mask, right_gt_mask_original == 255)
        
        # Calculate metrics using masks
        left_psnr, left_ssim, left_lpips = calculate_masked_metrics(
            left_img, left_gt_img, left_gt_mask_original, lpips_model)
        right_psnr, right_ssim, right_lpips = calculate_masked_metrics(
            right_img, right_gt_img, right_gt_mask_original, lpips_model)
        
        # Create color-coded visualizations using ORIGINAL masks
        create_color_visualization(
            left_proj_mask, 
            left_gt_mask_original == 255,
            left_vis_dir / f"{frame_name}_left_iou_vis.png"
        )
        create_color_visualization(
            right_proj_mask,
            right_gt_mask_original == 255, 
            right_vis_dir / f"{frame_name}_right_iou_vis.png"
        )
        
        # Store results
        all_results.append({
            'frame': frame_name,
            'left_iou': left_iou,
            'right_iou': right_iou,
            'left_psnr': left_psnr if left_psnr is not None else np.nan,
            'left_ssim': left_ssim if left_ssim is not None else np.nan,
            'left_lpips': left_lpips if left_lpips is not None else np.nan,
            'right_psnr': right_psnr if right_psnr is not None else np.nan,
            'right_ssim': right_ssim if right_ssim is not None else np.nan,
            'right_lpips': right_lpips if right_lpips is not None else np.nan
        })
        
        print(f"{frame_name} - Left IoU: {left_iou:.4f}, Right IoU: {right_iou:.4f}")
        if left_psnr is not None:
            print(f"  Left - PSNR: {left_psnr:.4f}, SSIM: {left_ssim:.4f}, LPIPS: {left_lpips:.4f}")
        if right_psnr is not None:
            print(f"  Right - PSNR: {right_psnr:.4f}, SSIM: {right_ssim:.4f}, LPIPS: {right_lpips:.4f}")
    
    # Save IoU scores to CSV
    results_df = pd.DataFrame(all_results)
    csv_path = left_output_dir.parent / 'reprojection_metrics.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to: {csv_path}")
    print(f"\nAverage Metrics:")
    print(f"Left  - IoU: {results_df['left_iou'].mean():.4f}, PSNR: {results_df['left_psnr'].mean():.4f}, SSIM: {results_df['left_ssim'].mean():.4f}, LPIPS: {results_df['left_lpips'].mean():.4f}")
    print(f"Right - IoU: {results_df['right_iou'].mean():.4f}, PSNR: {results_df['right_psnr'].mean():.4f}, SSIM: {results_df['right_ssim'].mean():.4f}, LPIPS: {results_df['right_lpips'].mean():.4f}")


if __name__ == "__main__":    main()