import cv2
import numpy as np
import os
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import json

# Try to import cupy for CUDA acceleration of numpy-like ops
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

def to_device_array(arr):
    """
    Converts a numpy array to cupy array if cupy is available, else returns numpy array.
    """
    if CUPY_AVAILABLE:
        return cp.asarray(arr)
    return arr

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
    Uses cupy for CUDA acceleration if available.
    """
    H, W = image_size
    arr_module = cp if CUPY_AVAILABLE else np
    img = arr_module.zeros((H, W, 3), dtype=arr_module.uint8)
    depth = arr_module.full((H, W), arr_module.inf, dtype=arr_module.float32)

    # Early exit if nothing valid
    if valid.sum() == 0:
        if CUPY_AVAILABLE:
            img = cp.asnumpy(img)
        return img

    # Prepare colors (RGB -> BGR) and move to array backend
    colors = rgb if rgb is not None else np.full((x.shape[0], 3), 255, dtype=np.uint8)
    colors_bgr = colors[:, [2, 1, 0]]
    colors_bgr_arr = arr_module.asarray(colors_bgr) if CUPY_AVAILABLE else colors_bgr

    # Select valid points
    xv = arr_module.asarray(x[valid]) if CUPY_AVAILABLE else x[valid]
    yv = arr_module.asarray(y[valid]) if CUPY_AVAILABLE else y[valid]
    zv = arr_module.asarray(z[valid]) if CUPY_AVAILABLE else z[valid]
    cv = colors_bgr_arr[valid]  # same backend as arr_module

    # Linear pixel index
    lin = yv * W + xv  # shape (N_valid,)

    # Sort by pixel, then by depth to keep nearest per pixel
    # argsort lexicographically: first by lin, then by zv
    # Create a key by stacking lin and zv
    if CUPY_AVAILABLE:
        # Cupy does not have lexsort; emulate by stable sort twice
        idx = cp.argsort(zv)           # sort by depth
        lin_s = lin[idx]
        zv_s = zv[idx]
        cv_s = cv[idx]
        idx2 = cp.argsort(lin_s, kind='stable')  # stable sort by pixel id
        lin_s = lin_s[idx2]
        zv_s = zv_s[idx2]
        cv_s = cv_s[idx2]
        # Unique pixels, take first occurrence (nearest depth due to prior sort)
        unique_lin, first_idx = cp.unique(lin_s, return_index=True)
        lin_k = lin_s[first_idx]
        zv_k = zv_s[first_idx]
        cv_k = cv_s[first_idx]
        yi_k = (lin_k // W).astype(cp.int32)
        xi_k = (lin_k % W).astype(cp.int32)
        # Assign in one shot
        depth[yi_k, xi_k] = zv_k
        img[yi_k, xi_k, :] = cv_k
        # Convert back to numpy for OpenCV ops
        img = cp.asnumpy(img)
    else:
        # Numpy path with lexsort on two keys (depth then pixel id)
        order = np.lexsort((zv, lin))  # primary key: lin, secondary: zv
        lin_s = lin[order]
        zv_s = zv[order]
        cv_s = cv[order]
        # Unique pixels, take first occurrence
        _, first_idx = np.unique(lin_s, return_index=True)
        lin_k = lin_s[first_idx]
        zv_k = zv_s[first_idx]
        cv_k = cv_s[first_idx]
        yi_k = (lin_k // W).astype(np.int32)
        xi_k = (lin_k % W).astype(np.int32)
        depth[yi_k, xi_k] = zv_k
        img[yi_k, xi_k, :] = cv_k

    # # Optional post-processing (on numpy)
    # if hole_filling:
    #     mask = (img.sum(axis=2) == 0).astype(np.uint8)
    #     if mask.sum() > 0:
    #         img = cv2.inpaint(img, mask, inpaintRadius=11, flags=cv2.INPAINT_TELEA)

    if hole_filling:
        mask = (img.sum(axis=2) == 0).astype(np.uint8)

        if mask.any():
            # 1. small dilation → erosion 填掉孔洞
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

            # 2. Optional: guided filter 让颜色过渡更自然
            try:
                import cv2.ximgproc as xip
                img = xip.guidedFilter(guide=closed, src=closed, radius=4, eps=1e-2)
            except:
                img = closed  # 没有 ximgproc 就只做 closing

    if gaussian_filter:
        img = cv2.GaussianBlur(img, (5, 5), 0.5)

    if interpolate:
        # Bilateral filter is expensive; keep parameters modest
        img = cv2.bilateralFilter(img, d=5, sigmaColor=20, sigmaSpace=5)

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

def overlay_tools_on_tissue(tissue_img, tools_img):
    """
    Overlays tools onto tissue by replacing non-zero tool pixels.
    Assumes both images are the same size and 3-channel BGR.
    """
    mask = (tools_img.sum(axis=2) > 0)
    out = tissue_img.copy()
    out[mask] = tools_img[mask]
    return out

def pair_ply_files_by_frame(tools_dir, tissue_dir):
    """
    Pair tool and tissue .ply files by matching frame numbers.
    Returns list of tuples (tools_ply, tissue_ply).
    """
    tools = [f for f in os.listdir(tools_dir) if f.lower().endswith(".ply")]
    tissue = [f for f in os.listdir(tissue_dir) if f.lower().endswith(".ply")]
    tools_map = {get_frame_number(f): f for f in tools if get_frame_number(f) is not None}
    tissue_map = {get_frame_number(f): f for f in tissue if get_frame_number(f) is not None}
    common_frames = sorted(set(tools_map.keys()) & set(tissue_map.keys()))
    pairs = [(tools_map[n], tissue_map[n]) for n in common_frames]
    return pairs

def find_corresponding_mask(rendered_filename, mask_dir):
    """
    Match mask image by frame number extracted from filenames.
    Accepts typical image extensions and prefers binary masks.
    """
    frame_num = get_frame_number(rendered_filename)
    candidates = [f for f in os.listdir(mask_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]
    if not candidates:
        return None
    candidates.sort()
    if frame_num is None:
        return candidates[0]
    matches = [f for f in candidates if get_frame_number(f) == frame_num]
    return matches[0] if matches else candidates[0]

def load_binary_mask(mask_path, target_size):
    """
    Loads a binary mask and resizes to target_size (H, W).
    Returns uint8 mask with values {0, 255}.
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    H, W = target_size
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    _, m_bin = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m_bin

def main():
    
    # pcd_path = "/workspace/EndoLRMGS/stereomis/p3/postprocessed_tools"
    # reference_path = "/workspace/datasets/endolrm_dataset/stereomis/p3/left_finalpass"
    # calib_path = "/workspace/datasets/endolrm_dataset/stereomis/p3/frame_data.json"
    # image_size = (1024, 1280)

    tools_pcd_path = "/workspace/EndoLRMGS/endonerf/pulling/postprocessed_tools"
    tissue_pcd_path = "/workspace/EndoLRMGS/endonerf/pulling/zxhezexin/openlrm-mix-base-1.1/tissue_reconstruction"
    reference_path = "/workspace/datasets/endolrm_dataset/endonerf/pulling/images"
    mask_path = "/workspace/datasets/endolrm_dataset/endonerf/pulling/binary_mask_deva"
    calib_path = "/workspace/datasets/endolrm_dataset/endonerf/pulling/frame_data.json"
    image_size = (512, 640)

    K, _ = load_calibration(calib_path)  # D ignored (images are calibrated)

    # Pair tool/tissue point clouds by frame number
    pairs = pair_ply_files_by_frame(tools_pcd_path, tissue_pcd_path)
    if not pairs:
        raise FileNotFoundError(f"No paired .ply files between {tools_pcd_path} and {tissue_pcd_path}")

    # LPIPS device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net='alex').to(device)

    # Accumulators
    psnr_list, ssim_list, lpips_list = [], [], []

    for tools_file, tissue_file in pairs:
        tools_path = os.path.join(tools_pcd_path, tools_file)
        tissue_path = os.path.join(tissue_pcd_path, tissue_file)

        # Load mask for this frame
        mask_name = find_corresponding_mask(tools_file, mask_path)
        mask_full_path = os.path.join(mask_path, mask_name) if mask_name else None
        mask_img = load_binary_mask(mask_full_path, image_size) if mask_full_path else None
        if mask_img is None:
            print(f"No mask for {tools_file}, proceeding without mask-constrained interpolation.")

        # Project tissue first (background)
        tissue_xyz, tissue_rgb = load_point_cloud(tissue_path)
        # Move arrays to GPU if cupy available
        tissue_xyz = to_device_array(tissue_xyz)
        tx, ty, tz, tvalid = project_points(tissue_xyz if not CUPY_AVAILABLE else cp.asnumpy(tissue_xyz), K, image_size)
        tissue_img = splat_points_to_image(
            tx, ty, tz, tvalid, tissue_rgb, image_size,
            interpolate=False, hole_filling=True, gaussian_filter=True
        )

        # Project tools second with interpolation
        tools_xyz, tools_rgb = load_point_cloud(tools_path)
        tools_xyz = to_device_array(tools_xyz)
        sx, sy, sz, svalid = project_points(tools_xyz if not CUPY_AVAILABLE else cp.asnumpy(tools_xyz), K, image_size)
        tools_img_raw = splat_points_to_image(
            sx, sy, sz, svalid, tools_rgb, image_size,
            interpolate=True, hole_filling=True, gaussian_filter=True
        )

        # Constrain interpolation to mask region:
        # - Create a filtered version (already interpolated)
        # - Combine filtered and raw outside mask to avoid bleeding
        if mask_img is not None:
            mask_bool = (mask_img > 0)
            tools_img = np.zeros_like(tools_img_raw)
            tools_img[mask_bool] = tools_img_raw[mask_bool]
            # ensure outside-mask pixels are zero to avoid overlay leaks
        else:
            tools_img = tools_img_raw

        # Overlay tools on tissue
        proj_img = overlay_tools_on_tissue(tissue_img, tools_img)

        # Debug outputs
        debug_path = os.path.join("projection.png")


        gt_name = find_corresponding_gt(tools_file, reference_path)
        if gt_name is None:
            print(f"No GT image found for {tools_file}, skipping.")
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
        # LPIPS on CUDA
        img1_rgb = cv2.cvtColor(gt_img_c, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(proj_img_c, cv2.COLOR_BGR2RGB)
        img1_t = lpips.im2tensor(img1_rgb).to(device)
        img2_t = lpips.im2tensor(img2_rgb).to(device)
        lpips_value = loss_fn(img1_t, img2_t).item()

        print(f"{tools_file} + {tissue_file} -> {gt_name}: PSNR {psnr_value:.4f}, SSIM {ssim_value:.4f}, LPIPS {lpips_value:.6f}")

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