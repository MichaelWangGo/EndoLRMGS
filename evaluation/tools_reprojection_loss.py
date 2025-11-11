import numpy as np
import cv2
import json
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from math import exp
import json
import lpips


@torch.no_grad()
def psnr(img1, img2, mask=None):
    if mask is None:
        mse_mask = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    else:
        if mask.shape[1] == 3:
            mse_mask = (((img1-img2)**2)*mask).sum() / ((mask.sum()+1e-10))
        else:
            mse_mask = (((img1-img2)**2)*mask).sum() / ((mask.sum()+1e-10)*3.0)

    return 20 * torch.log10(1.0 / torch.sqrt(mse_mask))

def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

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

def project_pointcloud_generic(point_cloud_path, K_or_P, image_size, R_rel=None, T_rel=None, output_path=None, interpolate=False, flip_z=False, use_projection_matrix=False, inpaint_near_points=False, near_dist_thresh=6):
    """Project point cloud into target view.
    
    Args:
        use_projection_matrix: If True, K_or_P is a 3x4 projection matrix; otherwise 3x3 intrinsic matrix
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

    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    depth_buffer = np.full((h, w), np.inf, dtype=np.float32)

    px = uv.astype(int)
    cols = (colors * 255).astype(np.uint8)

    for (x, y), c, d in zip(px, cols, depths):
        if d < depth_buffer[y, x]:
            img[y, x] = c
            depth_buffer[y, x] = d


    if interpolate:
        # Hole = no projection (depth_buffer == inf)
        hole_mask = (depth_buffer == np.inf)
        if hole_mask.any():
            if inpaint_near_points:
                # Distance (in pixels) from each hole to nearest valid pixel
                # distanceTransform expects 8-bit single channel: non-zero=foreground.
                # We give hole_mask to get distance inside holes.
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
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img

def compute_metrics_in_mask(pred_img, gt_img, mask, lpips_model):
    """Compute PSNR, SSIM, and LPIPS in masked region
    
    Args:
        pred_img: RGB image as numpy array (H, W, 3), float32, range [0, 255]
        gt_img: RGB image as numpy array (H, W, 3), float32, range [0, 255]
        mask: Binary mask as numpy array (H, W), bool
        lpips_model: Preloaded LPIPS model
    """
    if mask.sum() == 0:
        return None, None, None
    
    # Convert to torch tensors (1, 3, H, W), normalize to [0, 1]
    # Ensure input is uint8 for tf.to_tensor
    pred_tensor = torch.from_numpy(pred_img.astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    gt_tensor = torch.from_numpy(gt_img.astype(np.uint8)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    
    # Move to CUDA if available
    if torch.cuda.is_available():
        pred_tensor = pred_tensor.cuda()
        gt_tensor = gt_tensor.cuda()
        mask_tensor = mask_tensor.cuda()
    
    # PSNR - computed on masked region
    psnr_val = psnr(pred_tensor, gt_tensor, mask_tensor)
    
    # SSIM - computed on full image
    ssim_val = ssim(pred_tensor, gt_tensor)
    
    # LPIPS - normalize to [-1, 1] for LPIPS
    pred_normalized = pred_tensor * 2.0 - 1.0
    gt_normalized = gt_tensor * 2.0 - 1.0
    
    with torch.no_grad():
        lpips_val = lpips_loss(pred_normalized, gt_normalized, lpips_model)
    
    return psnr_val.item(), ssim_val.item(), lpips_val.item()

def main():
    # Paths
    pointcloud_dir = Path('/workspace/EndoLRMGS/stereomis/zxhezexin/ablation_study/base/postprocessed_tools')
    frame_data_file = Path('/workspace/datasets/endolrm_dataset/stereomis/p2_6/frame_data.json')
    right_mask_dir = Path('/workspace/Tracking-Anything-with-DEVA/output/stereomis/p2_6/right_finalpass/binary_mask_deva')
    right_gt_image_dir = Path('/workspace/datasets/endolrm_dataset/stereomis/p2_6/right_finalpass')
    left_mask_dir = Path('/workspace/datasets/endolrm_dataset/stereomis/p2_6/binary_mask_deva')
    left_gt_image_dir = Path('/workspace/datasets/endolrm_dataset/stereomis/p2_6/left_finalpass')
    left_output_dir = Path('/workspace/EndoLRMGS/stereomis/zxhezexin/ablation_study/base/left_view_reprojected_images')
    right_output_dir = Path('/workspace/EndoLRMGS/stereomis/zxhezexin/ablation_study/base/right_view_reprojected_images')
    left_output_dir.mkdir(exist_ok=True)
    right_output_dir.mkdir(exist_ok=True)

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

    for idx in tqdm(range(min_count)):
        pcd_path = point_clouds[idx]
        frame_name = pcd_path.stem

        # Left projection: retain original behavior
        left_img = project_pointcloud_generic(
            pcd_path, KL_orig, image_size, None, None,
            str(left_output_dir / f"{frame_name}_left.png"), interpolate=True, flip_z=False,
            inpaint_near_points=True, near_dist_thresh=5
        )

        # Right projection: use P2 directly (includes rectification offset)
        right_img = project_pointcloud_generic(
            pcd_path, P2, image_size, R_right, T_right,
            str(right_output_dir / f"{frame_name}_right.png"), 
            interpolate=True, use_projection_matrix=True, flip_z=False,
            inpaint_near_points=True, near_dist_thresh=5
        )

if __name__ == "__main__":
    main()