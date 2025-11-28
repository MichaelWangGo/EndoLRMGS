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


# ===== New code starts here =====

def load_calibration(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    cam = data["camera-calibration"]
    K = np.array(cam["KL"], dtype=np.float32)
    D = np.array(cam["DL"][0], dtype=np.float32).reshape(-1)  # k1,k2,p1,p2,k3
    # R_stereo = np.array(cam["R"], dtype=np.float32)           # not used for mono
    # T_stereo = np.array(cam["T"], dtype=np.float32).reshape(3)
    # pose = np.array(data["camera-pose"], dtype=np.float32)    # unused in projection
    return K, D

def load_point_cloud(ply_path):
    """
    Loads a point cloud from a specific .ply file.
    Tries open3d first, then plyfile. Returns (xyz: float32 Nx3, rgb: uint8 Nx3 or None).
    """
    # Try open3d
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        xyz = np.asarray(pcd.points, dtype=np.float32)
        rgb = None
        if pcd.has_colors():
            rgb = (np.clip(np.asarray(pcd.colors) * 255.0, 0, 255)).astype(np.uint8)  # RGB
        return xyz, rgb
    except Exception:
        pass
    # Fallback to plyfile
    try:
        from plyfile import PlyData
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        xyz = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=1).astype(np.float32)
        rgb = None
        if all(k in vertex.data.dtype.names for k in ('red', 'green', 'blue')):
            rgb = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=1).astype(np.uint8)  # RGB
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
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)  # no distortion
    img_pts = img_pts.reshape(-1, 2)
    # Filter valid depth and in-bounds
    H, W = image_size
    z = Pc[:, 2]
    valid = (z > 0)  # in front of camera
    x = np.round(img_pts[:, 0]).astype(np.int32)
    y = np.round(img_pts[:, 1]).astype(np.int32)
    valid &= (x >= 0) & (x < W) & (y >= 0) & (y < H)
    return x, y, z, valid


def splat_points_to_image(x, y, z, valid, rgb, image_size, interpolate=False, hole_filling=True, gaussian_filter=False):
    """
    Z-buffer splat of points into an image canvas with optional post-processing.
    Input rgb is expected in RGB order; convert to BGR for OpenCV.
    """
    H, W = image_size
    img = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), np.inf, dtype=np.float32)
    colors = rgb if rgb is not None else np.full((x.shape[0], 3), 255, dtype=np.uint8)
    colors_bgr = colors[:, [2, 1, 0]]

    # Z-buffer splatting with multi-sample per pixel (reduces aliasing)
    for i in np.where(valid)[0]:
        xi, yi = x[i], y[i]
        if z[i] < depth[yi, xi]:
            depth[yi, xi] = z[i]
            img[yi, xi] = colors_bgr[i]

    # Fill holes using inpainting
    if hole_filling:
        # import ipdb; ipdb.set_trace()
        mask = (img.sum(axis=2) == 0).astype(np.uint8)
        if mask.sum() > 0:
            img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Apply slight Gaussian smoothing to reduce noise
    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0.5)

    if interpolate:
        # Apply edge-preserving interpolation over the whole image
        img = cv2.bilateralFilter(img, d=9, sigmaColor=25, sigmaSpace=7)

    return img

# Add a helper to crop borders by a given fraction on all sides
def crop_border(img, frac=0.1):
    H, W = img.shape[:2]
    t = int(H * frac)
    b = H - int(H * frac)
    l = int(W * frac)
    r = W - int(W * frac)
    if b <= t or r <= l:
        return img
    return img[t:b, l:r]

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
    
    pcd_path = "/workspace/EndoLRMGS/stereomis/p3/final_results"
    reference_path = "/workspace/datasets/endolrm_dataset/stereomis/p3/left_finalpass"
    calib_path = "/workspace/datasets/endolrm_dataset/stereomis/p3/frame_data.json"
    image_size = (1024, 1280)

    K, _ = load_calibration(calib_path)  # D ignored (images are calibrated)
    # Assume target image size from principal point in K or known dataset size

    # Iterate all .ply point clouds
    ply_files = [f for f in os.listdir(pcd_path) if f.lower().endswith(".ply")]
    if not ply_files:
        raise FileNotFoundError(f"No .ply files in {pcd_path}")

    loss_fn = lpips.LPIPS(net='alex').cuda()

    # Accumulators
    psnr_list, ssim_list, lpips_list = [], [], []

    for ply_file in sorted(ply_files):
        ply_path = os.path.join(pcd_path, ply_file)
        xyz, rgb = load_point_cloud(ply_path)
        # Assume xyz already in camera frame; do NOT apply pose
        x, y, z, valid = project_points(xyz, K, image_size)
        # Enable hole filling and slight smoothing
        proj_img = splat_points_to_image(x, y, z, valid, rgb, image_size, 
                                         interpolate=False, 
                                         hole_filling=True, 
                                         gaussian_filter=True)
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

        # Crop 10% off each border for both images
        proj_img_c = crop_border(proj_img, frac=0.1)
        gt_img_c = crop_border(gt_img, frac=0.1)
        cv2.imwrite(debug_path, proj_img_c)
        cv2.imwrite(debug_path.replace("projection.png", "gt_cropped.png"), gt_img_c)
        # Metrics on cropped images
        psnr_value = psnr(gt_img_c, proj_img_c, data_range=255)
        ssim_value = ssim(cv2.cvtColor(gt_img_c, cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(proj_img_c, cv2.COLOR_BGR2GRAY),
                          data_range=255)
        img1_rgb = cv2.cvtColor(gt_img_c, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(proj_img_c, cv2.COLOR_BGR2RGB)
        img1_t = lpips.im2tensor(img1_rgb).cuda()
        img2_t = lpips.im2tensor(img2_rgb).cuda()
        lpips_value = loss_fn(img1_t, img2_t).item()

        print(f"{ply_file} -> {gt_name}: PSNR {psnr_value:.4f}, SSIM {ssim_value:.4f}, LPIPS {lpips_value:.6f}")

        # Accumulate
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
        lpips_list.append(lpips_value)

    # Averages
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