import cv2
import numpy as np
import os
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips
import torch


def tensor2array(tensor):
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    else:
        return tensor

# 加载图像
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"图像无法加载: {image_path}")
    return image

def load_and_resize_image(image_path, target_size=None):
    """Load image and optionally resize it to match target dimensions"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"图像无法加载: {image_path}")
    if target_size is not None and image.shape[:2] != target_size:
        image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    return image

def load_mask(mask_path):
    # import ipdb; ipdb.set_trace()
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # mask = 255 - mask # add this if evaluate tissue; or comment this line if evaluate tools
    if mask is None:
        raise ValueError(f"Mask cannot be loaded: {mask_path}")
    return mask > 0

def calculate_masked_metrics(image1, image2, mask):
    # Apply mask to both images
    masked_image1 = image1.copy()
    masked_image2 = image2.copy()
    # import ipdb; ipdb.set_trace()
    # Set masked areas to 0
    masked_image1 = masked_image1 * mask[:, :, np.newaxis]
    masked_image2 = masked_image2 * mask[:, :, np.newaxis]
    # masked_image2[~mask] = 0
    cv2.imwrite('masked_image1.png', masked_image1)
    # Calculate metrics only on unmasked areas
    psnr_value = psnr(masked_image1, masked_image2)
    
    gray1 = cv2.cvtColor(masked_image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(masked_image2, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(gray1, gray2)
    
    # For LPIPS, we need to handle the mask differently
    image1_tensor = lpips.im2tensor(masked_image1).cuda()
    image2_tensor = lpips.im2tensor(masked_image2).cuda()
    loss_fn = lpips.LPIPS(net='alex').cuda()
    lpips_value = loss_fn(image1_tensor, image2_tensor).item()
    
    return psnr_value, ssim_value, lpips_value

def cal_rmse(a, b, mask):
    """Compute rmse.
    """
    if torch.is_tensor(a):
        a = tensor2array(a)
    if torch.is_tensor(b):
        b = tensor2array(b)
    if torch.is_tensor(mask):
        mask = tensor2array(mask)
    if len(mask.shape) == len(a.shape) - 1:
        mask = mask[..., None]
    mask_sum = np.sum(mask) + 1e-10
    rmse = (((a - b)**2 * mask).sum() / (mask_sum))**0.5
    return rmse


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

def load_depth_image(depth_path):
    """Load depth image as single channel float32"""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Depth image cannot be loaded: {depth_path}")
    return depth.astype(np.float32)

# 主函数
def main():
    # Define paths
    rendered_dir = '/workspace/EndoLRM2/scared/zxhezexin/openlrm-mix-large-1.1/rendered'
    gt_dir = '/workspace/dataset/endolrm_dataset/scared/dataset_6/data_processed/rgb_scale_1'
    mask_dir = '/workspace/dataset/endolrm_dataset/scared/dataset_6/data_processed/mask_scale_1'
    rendered_depth = '/workspace/EndoLRM2/scared/zxhezexin/openlrm-mix-large-1.1/rendered_depth'
    gt_depth = '/workspace/Bidirectional-SemiSupervised-Dual-branch-CNN/scared_d6k4'
    
    rendered_files = sorted([f for f in os.listdir(rendered_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_depth_rmse = 0
    count = 0
    
    for rendered_file in rendered_files:
        gt_file, mask_file = find_corresponding_files(rendered_file, gt_dir, mask_dir)
        
        if gt_file is None or mask_file is None:
            print(f"Skipping {rendered_file}: No matching GT or mask file")
            continue
            
        rendered_path = os.path.join(rendered_dir, rendered_file)
        gt_path = os.path.join(gt_dir, gt_file)
        mask_path = os.path.join(mask_dir, mask_file)
        # import ipdb; ipdb.set_trace()
        # Get corresponding depth file paths
        depth_file = rendered_file.replace('_endo_render.png', '_endo_depth.png')
        depth_gt_file = rendered_file.replace('_endo_render.png', '.png')
        rendered_depth_path = os.path.join(rendered_depth, depth_file)
        gt_depth_path = os.path.join(gt_depth, depth_gt_file)
        
        try:
            # First load rendered image to get target size
            rendered_img = load_image(rendered_path)
            target_size = rendered_img.shape[:2]  # (height, width)
            
            # Load and resize GT image and mask to match rendered image size
            gt_img = load_image(gt_path)
            # gt_img = cv2.resize(gt_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            
            mask = load_mask(mask_path)
            
            mask = cv2.resize(mask.astype(np.uint8), (target_size[1], target_size[0]), 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            # rendered_img = rendered_img[:, 40:, :]  # Crop the rendered image
            # gt_img = gt_img[:, 40:, :]
            # mask = mask[:, 40:]

            # Load and evaluate depth images
            if os.path.exists(rendered_depth_path) and os.path.exists(gt_depth_path):
                rendered_depth_img = load_depth_image(rendered_depth_path)*3
                gt_depth_img = load_depth_image(gt_depth_path) / 256
                # import ipdb; ipdb.set_trace()
                depth_rmse = cal_rmse(rendered_depth_img, gt_depth_img, mask)
                total_depth_rmse += depth_rmse
                print(f"Depth RMSE: {depth_rmse:.6f}")
            
            psnr_value, ssim_value, lpips_value = calculate_masked_metrics(
                rendered_img, gt_img, mask)
            
            print(f"Processing:")
            print(f"  Rendered: {rendered_file}")
            print(f"  GT: {gt_file}")
            print(f"  Mask: {mask_file}")
            print(f"PSNR: {psnr_value:.6f} dB")
            print(f"SSIM: {ssim_value:.6f}")
            print(f"LPIPS: {lpips_value:.6f}\n")
            
            total_psnr += psnr_value
            total_ssim += ssim_value
            total_lpips += lpips_value
            count += 1
            
        except Exception as e:
            print(f"Error processing {rendered_file}: {str(e)}")
            continue
    
    if count > 0:
        print("\nAverage metrics:")
        print(f"PSNR: {total_psnr/count:.6f} dB")
        print(f"SSIM: {total_ssim/count:.6f}")
        print(f"LPIPS: {total_lpips/count:.6f}")
        print(f"Depth RMSE: {total_depth_rmse/count:.6f}")

if __name__ == "__main__":
    main()