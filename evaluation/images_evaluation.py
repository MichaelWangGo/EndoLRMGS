import cv2
import numpy as np
import os
import re
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

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
    mask = 255 - mask # add this if evaluate tissue; or comment this line if evaluate tools
    if mask is None:
        raise ValueError(f"Mask cannot be loaded: {mask_path}")
    return mask > 0

def calculate_masked_metrics(image1, image2, mask):
    # Apply mask to both images
    masked_image1 = image1.copy()
    masked_image2 = image2.copy()
    # import ipdb; ipdb.set_trace()
    # Set masked areas to 0
    masked_image1[~mask] = 0
    masked_image2[~mask] = 0
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

# 主函数
def main():
    # Define paths
    rendered_dir = '/workspace/EndoLRM2/stereomis/zxhezexin/openlrm-mix-base-1.1/rendered'
    gt_dir = '/workspace/dataset/endolrm_dataset/stereomis/left_finalpass'
    mask_dir = '/workspace/dataset/endolrm_dataset/stereomis/binary_mask_deva'
    
    rendered_files = sorted([f for f in os.listdir(rendered_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))])
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    count = 0
    
    for rendered_file in rendered_files:
        gt_file, mask_file = find_corresponding_files(rendered_file, gt_dir, mask_dir)
        
        if gt_file is None or mask_file is None:
            print(f"Skipping {rendered_file}: No matching GT or mask file")
            continue
            
        rendered_path = os.path.join(rendered_dir, rendered_file)
        gt_path = os.path.join(gt_dir, gt_file)
        mask_path = os.path.join(mask_dir, mask_file)
        
        try:
            # First load rendered image to get target size
            rendered_img = load_image(rendered_path)
            target_size = rendered_img.shape[:2]  # (height, width)
            
            # Load and resize GT image and mask to match rendered image size
            gt_img = load_image(gt_path)
            gt_img = cv2.resize(gt_img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            
            mask = load_mask(mask_path)
            
            mask = cv2.resize(mask.astype(np.uint8), (target_size[1], target_size[0]), 
                            interpolation=cv2.INTER_NEAREST).astype(bool)
            
            psnr_value, ssim_value, lpips_value = calculate_masked_metrics(
                rendered_img, gt_img, mask)
            
            print(f"Processing:")
            print(f"  Rendered: {rendered_file}")
            print(f"  GT: {gt_file}")
            print(f"  Mask: {mask_file}")
            print(f"PSNR: {psnr_value:.4f} dB")
            print(f"SSIM: {ssim_value:.4f}")
            print(f"LPIPS: {lpips_value:.4f}\n")
            
            total_psnr += psnr_value
            total_ssim += ssim_value
            total_lpips += lpips_value
            count += 1
            
        except Exception as e:
            print(f"Error processing {rendered_file}: {str(e)}")
            continue
    
    if count > 0:
        print("\nAverage metrics:")
        print(f"PSNR: {total_psnr/count:.4f} dB")
        print(f"SSIM: {total_ssim/count:.4f}")
        print(f"LPIPS: {total_lpips/count:.4f}")

if __name__ == "__main__":
    main()