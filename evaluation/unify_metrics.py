import cv2
import numpy as np
import os
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import json

def get_frame_number(filename):
    # Extract frame number from filename using regex
    # Handles patterns like "frame_0001.png" or "0001.png"
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def find_corresponding_files(rendered_file, gt_dir, mask_dir):
    frame_num = get_frame_number(rendered_file)
    if frame_num is None:
        return None, None
    
    # Find corresponding GT file
    gt_files = [f for f in os.listdir(gt_dir) if get_frame_number(f) == frame_num]
    if not gt_files:
        return None, None
    
    # Find corresponding mask file
    mask_files = [f for f in os.listdir(mask_dir) if get_frame_number(f) == frame_num]
    if not mask_files:
        return None, None
    
    return gt_files[0], mask_files[0]

def load_calibration(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    cam = data["camera-calibration"]
    K = np.array(cam["KL"], dtype=np.float32)
    D = np.array(cam["DL"][0], dtype=np.float32).reshape(-1)
    return K, D

def load_point_cloud(ply_path):
    """
    Loads a point cloud from a specific .ply file.
    Tries open3d first, then plyfile. Returns (xyz: float32 Nx3, rgb: uint8 Nx3 or None).
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = None
        if pcd.has_colors():
            rgb = (np.clip(np.asarray(pcd.colors) * 255.0, 0, 255)).astype(np.uint8)
        return xyz, rgb
    except Exception:
        pass
    
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
        rgb = None
        if all(k in vertex.data.dtype.names for k in ('red', 'green', 'blue')):
            rgb = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1).astype(np.uint8)
        return xyz, rgb
    except Exception as e:
        raise RuntimeError(
            f"Failed to read {ply_path}. Install open3d (`pip install open3d`) or plyfile (`pip install plyfile`). Original error: {e}"
        )

def project_points(Pc, K, image_size):
    """
    Projects 3D camera-frame points using cv2.projectPoints.
    Distortion is ignored (images already calibrated).
    """
    obj_pts = Pc.reshape(-1, 1, 3).astype(np.float32)
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
    img_pts = img_pts.reshape(-1, 2)
    
    H, W = image_size
    z = Pc[:, 2]
    valid = (z > 0)
    x = np.round(img_pts[:, 0]).astype(np.int32)
    y = np.round(img_pts[:, 1]).astype(np.int32)
    valid &= (x >= 0) & (x < W) & (y >= 0) & (y < H)
    return x, y, z, valid

def determine_foreground_points(x, y, z, valid, image_size):
    """
    For each pixel, if multiple points project onto it, select the single closest (minimum depth) point as foreground.
    Returns:
      fg_point_mask: boolean array (N,), True for points that are closest at their pixel among all hits
      counts_map: HxW int map of hit counts per pixel
    """
    H, W = image_size
    N = x.shape[0]
    fg_point_mask = np.zeros(N, dtype=bool)
    if not np.any(valid):
        return fg_point_mask, np.zeros((H, W), dtype=np.int32)

    # Linear pixel indices for valid points
    lin_idx = (y.astype(np.int64) * W + x.astype(np.int64))
    idx_valid = np.where(valid)[0]
    lin_idx_valid = lin_idx[valid]
    z_valid = z[valid]

    # Group by pixel: find argmin depth per pixel among valid points
    # Use a dict of pixel -> (best_depth, best_global_idx)
    best = {}
    counts = np.bincount(lin_idx_valid, minlength=H * W)
    for gi, pidx, zi in zip(idx_valid, lin_idx_valid, z_valid):
        bd, bi = best.get(pidx, (np.inf, -1))
        if zi < bd:
            best[pidx] = (zi, gi)

    # Mark only the closest points per pixel as foreground, but only for pixels with 2+ hits
    multi_hit_pixels = np.where(counts > 1)[0]
    for pidx in multi_hit_pixels:
        if pidx in best:
            _, gi = best[pidx]
            fg_point_mask[gi] = True

    return fg_point_mask, counts.reshape(H, W)

def render_with_foreground_coverage(x, y, z, valid, rgb, image_size,
                                    gaussian_filter=True, closing_kernel=(9, 9), closing_iters=2, inpaint_radius=5):
    """
    Two-pass render ensuring the foreground (closest points per pixel where multiple depths exist) fully covers background:
      1) Render all points: background layer + vis_mask_all (no filling).
      2) Detect foreground points as closest per pixel where multiple hits exist, render foreground-only.
      3) Build a solid foreground coverage mask by morphological closing + contour fill.
      4) Inpaint ALL missing pixels inside the foreground coverage mask.
      5) Composite foreground over background.
    Returns:
      final_img (BGR uint8), min_depth_map (float32), vis_mask_all (bool)
    """
    # 1) Background/raw layer (no hole filling so vis_mask only marks true projected points)
    bg_img, min_depth_map, vis_mask_all = splat_points_to_image(
        x, y, z, valid, rgb, image_size,
        interpolate=False, hole_filling=False, gaussian_filter=False
    )

    # 2) Foreground detection: closest points per pixel among multi-hit pixels
    fg_pts_mask, _ = determine_foreground_points(x, y, z, valid, image_size)
    sel = valid & fg_pts_mask
    if np.any(sel):
        x_fg, y_fg, z_fg = x[sel], y[sel], z[sel]
        rgb_fg = rgb[sel] if rgb is not None else None

        # Foreground-only render (no filling here to preserve true vis for mask building)
        fg_img, _, fg_vis = splat_points_to_image(
            x_fg, y_fg, z_fg, np.ones_like(x_fg, dtype=bool), rgb_fg, image_size,
            interpolate=False, hole_filling=False, gaussian_filter=False
        )

        # 3) Solid foreground coverage mask
        fg_mask = fg_vis.astype(np.uint8)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, closing_kernel)
        fg_closed = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=closing_iters)
        contours, _ = cv2.findContours(fg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fg_full_mask = np.zeros_like(fg_closed, dtype=np.uint8)
        if len(contours) > 0:
            cv2.drawContours(fg_full_mask, contours, -1, color=255, thickness=cv2.FILLED)
        fg_full_mask_bool = fg_full_mask.astype(bool)

        # 4) Inpaint ALL missing pixels inside the full foreground mask
        holes_in_fg_full = fg_full_mask_bool & (fg_img.sum(axis=2) == 0)
        if np.any(holes_in_fg_full):
            inpaint_mask = holes_in_fg_full.astype(np.uint8)
            fg_img = cv2.inpaint(fg_img, inpaint_mask, inpaint_radius, flags=cv2.INPAINT_TELEA)
        # Optional: edge-preserving interpolation to refine filled areas
        fg_img = cv2.bilateralFilter(fg_img, d=9, sigmaColor=25, sigmaSpace=7)

        # 5) Composite: foreground overrides background wherever fg_full_mask is true
        final_img = bg_img.copy()
        final_img[fg_full_mask_bool] = fg_img[fg_full_mask_bool]
    else:
        # No multi-depth pixels; background-only render is sufficient
        final_img = bg_img

    if gaussian_filter:
        final_img = cv2.GaussianBlur(final_img, (5, 5), 0.5)

    return final_img, min_depth_map, vis_mask_all

def splat_points_to_image(x, y, z, valid, rgb, image_size, interpolate=False, hole_filling=True, gaussian_filter=False):
    """
    Z-buffer splat of points into an image canvas with optional post-processing.
    Input rgb is expected in RGB order; convert to BGR for OpenCV.
    Returns:
      img: rendered BGR image
      depth: min depth map (float32), np.inf for pixels with no points
      vis_mask: boolean visibility mask (True = actual rendered point, before hole filling)
    """
    H, W = image_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), np.inf, dtype=np.float32)
    vis_mask_original = np.zeros((H, W), dtype=bool)
    colors = rgb if rgb is not None else np.full((x.shape[0], 3), 255, dtype=np.uint8)
    colors_bgr = colors[:, [2, 1, 0]]

    # Z-buffer: keep only closest per pixel
    for i in np.where(valid)[0]:
        xi, yi = x[i], y[i]
        zi = z[i]
        if zi < depth[yi, xi]:
            depth[yi, xi] = zi
            img[yi, xi] = colors_bgr[i]
            vis_mask_original[yi, xi] = True

    vis_mask = vis_mask_original.copy()

    # Optional generic hole filling (not foreground-specific)
    if hole_filling:
        mask = (img.sum(axis=2) == 0).astype(np.uint8)
        if mask.sum() > 0:
            img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0.5)

    if interpolate:
        img = cv2.bilateralFilter(img, d=9, sigmaColor=25, sigmaSpace=7)

    return img, depth, vis_mask

def crop_border(img, frac=0.1):
    H, W = img.shape[:2]
    t = int(H * frac)
    b = H - int(H * frac)
    l = int(W * frac)
    r = W - int(W * frac)
    if b <= t or r <= l:
        return img
    return img[t:b, l:r]

def crop_mask(mask, frac=0.1):
    H, W = mask.shape[:2]
    t = int(H * frac)
    b = H - int(H * frac)
    l = int(W * frac)
    r = W - int(W * frac)
    if b <= t or r <= l:
        return mask
    return mask[t:b, l:r]

def masked_psnr(img_ref, img_pred, mask):
    """
    Compute PSNR only over pixels where mask is True.
    img_ref, img_pred: uint8 BGR images
    mask: boolean 2D array
    """
    if mask.sum() == 0:
        return float('nan')
    ref = img_ref.reshape(-1, 3)[mask.reshape(-1)]
    pred = img_pred.reshape(-1, 3)[mask.reshape(-1)]
    mse = np.mean((ref.astype(np.float32) - pred.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    PIX_MAX = 255.0
    return 20.0 * np.log10(PIX_MAX) - 10.0 * np.log10(mse)

def masked_ssim_global(img_ref_gray, img_pred_gray, mask):
    """
    Compute a global SSIM (single-window) over masked pixels using the standard SSIM formula.
    This is not the sliding-window SSIM, but respects the vis_mask region.
    """
    m = mask.reshape(-1)
    if m.sum() == 0:
        return float('nan')
    x = img_ref_gray.reshape(-1)[m].astype(np.float64)
    y = img_pred_gray.reshape(-1)[m].astype(np.float64)
    
    L = 255.0
    K1, K2 = 0.01, 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x2 = x.var(ddof=1) if x.size > 1 else 0.0
    sigma_y2 = y.var(ddof=1) if y.size > 1 else 0.0
    sigma_xy = np.cov(x, y, ddof=1)[0, 1] if x.size > 1 else 0.0
    
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(num / den) if den != 0 else float('nan')

def find_corresponding_gt(rendered_filename, gt_dir):
    """
    Match reference image (.png) by frame number extracted from filenames.
    Falls back to the first sorted PNG if no numeric match is found.
    """
    frame_num = get_frame_number(rendered_filename)
    candidates = [f for f in os.listdir(gt_dir) if f.lower().endswith(".png")]
    if not candidates:
        return None
    candidates.sort()
    if frame_num is None:
        return candidates[0]
    matches = [f for f in candidates if get_frame_number(f) == frame_num]
    return matches[0] if matches else candidates[0]

def main():
    # pcd_path = "/workspace/EndoLRMGS/stereomis/p3/final_results"
    # reference_path = "/workspace/datasets/endolrm_dataset/stereomis/p3/left_finalpass"
    # calib_path = "/workspace/datasets/endolrm_dataset/stereomis/p3/frame_data.json"
    # image_size = (1024, 1280)

    pcd_path = "/workspace/EndoLRMGS/endonerf/pulling/final_results"
    reference_path = "/workspace/datasets/endolrm_dataset/endonerf/pulling/images"
    calib_path = "/workspace/datasets/endolrm_dataset/endonerf/pulling/frame_data.json"
    image_size = (512, 640)

    K, _ = load_calibration(calib_path)

    ply_files = [f for f in os.listdir(pcd_path) if f.lower().endswith(".ply")]
    if not ply_files:
        raise FileNotFoundError(f"No .ply files in {pcd_path}")

    loss_fn = lpips.LPIPS(net='alex').cuda()

    psnr_list, ssim_list, lpips_list = [], [], []

    for ply_file in sorted(ply_files):
        ply_path = os.path.join(pcd_path, ply_file)
        xyz, rgb = load_point_cloud(ply_path)
        x, y, z, valid = project_points(xyz, K, image_size)

        # Enforce foreground coverage using closest-depth selection per pixel
        proj_img, min_depth_map, vis_mask = render_with_foreground_coverage(
            x, y, z, valid, rgb, image_size,
            gaussian_filter=True, closing_kernel=(3, 3), closing_iters=2, inpaint_radius=9
        )

        debug_path = os.path.join("projection.png")

        gt_name = find_corresponding_gt(ply_file, reference_path)
        if gt_name is None:
            print(f"No GT image found for {ply_file}, skipping.")
            continue
        
        gt_img = cv2.imread(os.path.join(reference_path, gt_name), cv2.IMREAD_COLOR)
        if gt_img is None:
            print(f"Failed to load GT {gt_name}, skipping.")
            continue

        if proj_img.shape != gt_img.shape:
            gt_img = cv2.resize(gt_img, (proj_img.shape[1], proj_img.shape[0]), interpolation=cv2.INTER_AREA)
        gt_img = cv2.GaussianBlur(gt_img, (5, 5), 0.5)

        # Crop 10% off each border for both images and mask
        proj_img_c = crop_border(proj_img, frac=0.1)
        gt_img_c = crop_border(gt_img, frac=0.1)
        vis_mask_c = crop_mask(vis_mask, frac=0.1)

        cv2.imwrite(debug_path, proj_img_c)
        cv2.imwrite(debug_path.replace("projection.png", "gt_cropped.png"), gt_img_c)

        # Metrics computed only on visible pixels
        psnr_value = masked_psnr(gt_img_c, proj_img_c, vis_mask_c)

        gt_gray_c = cv2.cvtColor(gt_img_c, cv2.COLOR_BGR2GRAY)
        proj_gray_c = cv2.cvtColor(proj_img_c, cv2.COLOR_BGR2GRAY)
        ssim_value = masked_ssim_global(gt_gray_c, proj_gray_c, vis_mask_c)

        # LPIPS over visible region: set non-visible pixels equal in both to neutralize influence
        img1_rgb = cv2.cvtColor(gt_img_c, cv2.COLOR_BGR2RGB).astype(np.uint8)
        img2_rgb = cv2.cvtColor(proj_img_c, cv2.COLOR_BGR2RGB).astype(np.uint8)

        # import ipdb; ipdb.set_trace()
        inv_mask = ~vis_mask_c
        if inv_mask.any():
            img2_rgb[inv_mask] = img1_rgb[inv_mask]
        
        img1_t = lpips.im2tensor(img1_rgb).cuda()
        img2_t = lpips.im2tensor(img2_rgb).cuda()
        lpips_value = loss_fn(img1_t, img2_t).item()

        print(f"{ply_file} -> {gt_name}: PSNR {psnr_value:.4f}, SSIM {ssim_value:.4f}, LPIPS {lpips_value:.6f}")

        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
        lpips_list.append(lpips_value)

    if psnr_list:
        avg_psnr = float(np.mean(psnr_list))
        avg_ssim = float(np.mean(ssim_list))
        avg_lpips = float(np.mean(lpips_list))
        print(f"Average metrics over {len(psnr_list)} pairs: "
              f"PSNR {avg_psnr:.4f}, SSIM {avg_ssim:.4f}, LPIPS {avg_lpips:.6f}")
    else:
        print("No valid pairs processed. No average metrics available.")

if __name__ == "__main__":
    main()