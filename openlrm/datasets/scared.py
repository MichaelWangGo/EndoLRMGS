# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Union
import random
import numpy as np
import torch
import json
from megfile import smart_path_join, smart_open

from .base import BaseDataset
from .cam_utils import build_camera_standard, build_camera_principle, camera_normalization_objaverse
from ..utils.proxy import no_proxy

__all__ = ['ScaredDataset']

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def extract_r_t(json_data):
    R = torch.tensor(json_data["R"]["data"], dtype=torch.float32).view(3, 3)
    T = torch.tensor(json_data["T"]["data"], dtype=torch.float32).view(3, 1)
    return R, T

def create_pose_matrix(R, T):
    pose = torch.cat((R, T), dim=1)  # 合并 R 和 T，形成 3x4 矩阵
    bottom_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)  # 添加 [0, 0, 0, 1]
    pose = torch.cat((pose, bottom_row), dim=0)  # 合并为 4x4 矩阵
    return pose

class ScaredDataset(BaseDataset):

    def __init__(self, root_dirs: list[str], meta_path: str,
                 sample_side_views: int,
                 render_image_res_low: int, render_image_res_high: int, render_region_size: int,
                 source_image_res: int, normalize_camera: bool,
                 normed_dist_to_center: Union[float, str] = None, num_all_views: int = 32):
        super().__init__(root_dirs, meta_path)
        self.sample_side_views = sample_side_views
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high
        self.render_region_size = render_region_size
        self.source_image_res = source_image_res
        self.normalize_camera = normalize_camera
        self.normed_dist_to_center = normed_dist_to_center
        self.num_all_views = num_all_views

    @staticmethod
    def _load_pose(file_path):
        pose = np.load(smart_open(file_path, 'rb'))
        pose = torch.from_numpy(pose).float()
        return pose

    @no_proxy
    def inner_get_item(self, idx):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """
        uid = self.uids[idx]
        current_dir = os.path.join(self.root_dirs[0], uid)
        
        pose_dir = os.path.join(current_dir, 'pose')
        rgb_dir = os.path.join(current_dir, 'rgb')
        # intrinsics_path = os.path.join(root_dir, uid, 'intrinsics.json')

        # # load intrinsics
        # intrinsics = np.load(smart_open(intrinsics_path, 'rb'))
        # intrinsics = torch.from_numpy(intrinsics).float()

        # sample views (incl. source view and side views)
        sample_views = np.random.choice(range(self.num_all_views), self.sample_side_views + 1, replace=False)
        # poses, rgbs, bg_colors = [], [], []
        bg_color = random.choice([0.0, 0.5, 1.0])
        # source_image = None
        source_path = os.path.join(current_dir, 'rgb/left.png')
        render_path = os.path.join(current_dir, 'rgb/right.png')
        source_image = self._load_rgb_image(source_path, bg_color= bg_color)
        render_image = self._load_rgb_image(render_path, bg_color= bg_color)
        import ipdb; ipdb.set_trace()
        "此处render_camera还需要将intrinsics加入; 检查source_camera的16是否为extrinsic和intrinsic的结合"
        # depth_map = 

        # pose_path = smart_path_join(pose_dir, f'{view:03d}.npy')
        # source_pose = self._load_pose(pose_path)
        # source_camera = build_camera_principle(poses[:1], intrinsics.unsqueeze(0)).squeeze(0)
        source_camera = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]]).view(1, 16).squeeze(0)
        
        calib_file = load_json(os.path.join(current_dir, 'pose/stereo_calib.json'))
        R, T = extract_r_t(calib_file)
        render_camera = create_pose_matrix(R, T).view(1, 16).squeeze(0)
        print("source_camera", source_camera)
        print("render_camera", render_camera)
        print("uid", uid)

        
        # adjust source image resolution
        source_image = torch.nn.functional.interpolate(
            source_image, size=(self.source_image_res, self.source_image_res), mode='bicubic', align_corners=True).squeeze(0)
        source_image = torch.clamp(source_image, 0, 1)

        # adjust render image resolution and sample intended rendering region
        render_image_res = np.random.randint(self.render_image_res_low, self.render_image_res_high + 1)
        render_image = torch.nn.functional.interpolate(
            rgbs, size=(render_image_res, render_image_res), mode='bicubic', align_corners=True)
        render_image = torch.clamp(render_image, 0, 1)
        anchors = torch.randint(
            0, render_image_res - self.render_region_size + 1, size=(self.sample_side_views + 1, 2))
        crop_indices = torch.arange(0, self.render_region_size, device=render_image.device)
        index_i = (anchors[:, 0].unsqueeze(1) + crop_indices).view(-1, self.render_region_size, 1)
        index_j = (anchors[:, 1].unsqueeze(1) + crop_indices).view(-1, 1, self.render_region_size)
        batch_indices = torch.arange(self.sample_side_views + 1, device=render_image.device).view(-1, 1, 1)
        cropped_render_image = render_image[batch_indices, :, index_i, index_j].permute(0, 3, 1, 2)

        return {
            'source_camera': source_camera,
            'render_camera': render_camera,
            'source_image': source_image,
            'render_image': cropped_render_image,
            'depth_map':depth_map,
            'render_anchors': anchors,
            'render_full_resolutions': torch.tensor([[render_image_res]], dtype=torch.float32).repeat(self.sample_side_views + 1, 1),
            'render_bg_colors': torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1),
        }
