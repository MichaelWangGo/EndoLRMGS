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


import torch
import os
import argparse
import mcubes
import trimesh
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from datetime import datetime
from pathlib import Path
from accelerate.logging import get_logger
from argparse import ArgumentParser
import torchvision
import cv2
import open3d as o3d
import time
from collections import defaultdict

from .base_inferrer import Inferrer
from openlrm.datasets.cam_utils import build_camera_principle, build_camera_standard, surrounding_views_linspace, create_intrinsics
from openlrm.utils.logging import configure_logger
from openlrm.runners import REGISTRY_RUNNERS
# from openlrm.utils.video import images_to_video
from openlrm.utils.hf_hub import wrap_model_hub
from EndoGaussian.scene import Scene, GaussianModel
from EndoGaussian.gaussian_renderer import render
from EndoGaussian.arguments import ModelParams, PipelineParams, ModelHiddenParams
# from ..gaussian_args import create_gaussian_args
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = get_logger(__name__)


def parse_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--infer', type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get('APP_INFER') is not None:
        args.infer = os.environ.get('APP_INFER')
    if os.environ.get('APP_MODEL_NAME') is not None:
        cli_cfg.model_name = os.environ.get('APP_MODEL_NAME')

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(cfg_train.experiment.parent, cfg_train.experiment.child, os.path.basename(cli_cfg.model_name).split('_')[-1])
        cfg.video_dump = os.path.join("exps", 'videos', _relative_path)
        cfg.mesh_dump = os.path.join("exps", 'meshes', _relative_path)
        cfg.rendered_dump = os.path.join("exps", 'rendered', _relative_path)
        cfg.rendered_depth = os.path.join("exps", 'rendered_depth', _relative_path)

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault('video_dump', os.path.join(str(cfg_infer.save_path), cli_cfg.model_name, 'videos'))
        cfg.setdefault('mesh_dump', os.path.join(str(cfg_infer.save_path), cli_cfg.model_name, 'meshes'))
        cfg.setdefault('rendered_dump', os.path.join(str(cfg_infer.save_path), cli_cfg.model_name, 'rendered'))
        cfg.setdefault('rendered_depth', os.path.join(str(cfg_infer.save_path), cli_cfg.model_name, 'rendered_depth'))

    cfg.merge_with(cli_cfg)

    """
    [required]
    model_name: str
    image_input: str
    export_video: bool
    export_mesh: bool

    [special]
    source_size: int
    render_size: int
    video_dump: str
    mesh_dump: str

    [default]
    render_views: int
    render_fps: int
    mesh_size: int
    mesh_thres: float
    frame_size: int
    logger: str
    """

    cfg.setdefault('logger', 'INFO')

    # assert not (args.config is not None and args.infer is not None), "Only one of config and infer should be provided"
    assert cfg.model_name is not None, "model_name is required"
    if not os.environ.get('APP_ENABLED', None):
        assert cfg.image_input is not None, "image_input is required"
        # assert cfg.export_video or cfg.export_mesh, \
        #     "At least one of export_video or export_mesh should be True"
        cfg.app_enabled = False
    else:
        cfg.app_enabled = True

    return cfg


@REGISTRY_RUNNERS.register('infer.lrm')
class LRMInferrer(Inferrer):

    EXP_TYPE: str = 'lrm'

    def __init__(self, freeze_endo_gaussian=True, gaussian_config=None):
        super().__init__(freeze_endo_gaussian)
        self.gaussian_config = gaussian_config
        self.timing_stats = defaultdict(float)  # Add timing stats dictionary
        self.num_processed = 0  # Counter for processed images
        
        self.cfg = parse_configs()
        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )
        
        # Initialize EndoGaussian components first
        self.scene = None
        self.gaussians = None
        self.pipe = None
        self.background = None
        self._setup_endogaussian()
        
        # Build LRM model
        self.model = self._build_model(self.cfg).to(self.device)
        logger.info("LRM model loaded and ready for inference")

    def _setup_endogaussian(self):
        """Initialize EndoGaussian model and components"""
        parser = ArgumentParser(description="Inference parameters")
        lp = ModelParams(parser, sentinel=True)
        pp = PipelineParams(parser)
        hp = ModelHiddenParams(parser)

        # Load config
        import mmcv
        config = mmcv.Config.fromfile(self.gaussian_config)
        args = parser.parse_args([])
        
        # Update args from config
        for k, v in config.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    setattr(args, sub_k, sub_v)

        # Extract parameters
        dataset = lp.extract(args)
        hyper = hp.extract(args)
        self.pipe = pp.extract(args)
        
        # Setup model components
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.gaussians = GaussianModel(dataset.sh_degree, hyper)
        self.scene = Scene(dataset, self.gaussians, load_iteration=args.iterations)

    def _get_endogaussian_render(self, idx):
        """Get rendered image from EndoGaussian"""
        viewpoint_cam = self.scene.getVideoCameras()[idx]
        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, self.background, stage="fine")
        
        rendered_image = render_pkg["render"]
        rendered_depth = render_pkg["depth"]
        rendered_depth = torch.clamp(rendered_depth, 0, 255).cpu()
        
        # Adjust dimensions to multiple of 14 for compatibility
        # H, W = rendered_image.shape[-2:]
        # new_H = ((H + 13) // 14) * 14
        # new_W = ((W + 13) // 14) * 14
        # rendered_image = torch.nn.functional.interpolate(
        #     rendered_image.unsqueeze(0), 
        #     size=(new_H, new_W),
        #     mode='bilinear',
        #     align_corners=True
        # )
        
        # rendered_depth = torch.nn.functional.interpolate(
        #     rendered_depth.unsqueeze(0),
        #     size=(new_H, new_W), 
        #     mode='bilinear',
        #     align_corners=True
        # )
        
        return rendered_image, rendered_depth

    def _default_source_camera(self, dist_to_center: float = 2.0, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -dist_to_center],
            [0, 1, 0, 0],
        ]], dtype=torch.float32, device=device)
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(self, n_views: int, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(n_views=n_views, device=device)
        render_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)
        render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    def infer_planes(self, image: torch.Tensor, source_cam_dist: float):

        N = image.shape[0]
        source_camera = self._default_source_camera(dist_to_center=source_cam_dist, batch_size=N, device=self.device)
        planes = self.model.forward_planes(image, source_camera)
        assert N == planes.shape[0]
        return planes

    def infer_video(self, planes: torch.Tensor,  frame_size: int, render_size: int, render_views: int, render_fps: int):
        N = planes.shape[0]
        render_cameras = self._default_render_cameras(n_views=render_views, batch_size=N, device=self.device)
        render_anchors = torch.zeros(N, render_cameras.shape[1], 2, device=self.device)
        render_resolutions = torch.ones(N, render_cameras.shape[1], 1, device=self.device) * render_size
        render_bg_colors = torch.ones(N, render_cameras.shape[1], 1, device=self.device, dtype=torch.float32) * 1.

        frames = []
        for i in range(0, render_cameras.shape[1], frame_size):
            frames.append(
                self.model.synthesizer(
                    planes=planes,
                    cameras=render_cameras[:, i:i+frame_size],
                    anchors=render_anchors[:, i:i+frame_size],
                    resolutions=render_resolutions[:, i:i+frame_size],
                    bg_colors=render_bg_colors[:, i:i+frame_size],
                    region_size=render_size,
                )
            )
        # merge frames
        frames = {
            k: torch.cat([r[k] for r in frames], dim=1)
            for k in frames[0].keys()
        }
        # dump
        # os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        # for k, v in frames.items():
        #     if k == 'images_rgb':
        #         images_to_video(
        #             images=v[0],
        #             output_path=dump_video_path,
        #             fps=render_fps,
        #             gradio_codec=self.cfg.app_enabled,
        #         )

    def infer_mesh(self, planes: torch.Tensor, mesh_size: int, mesh_thres: float, dump_mesh_path: str):
        grid_out = self.model.synthesizer.forward_grid(
            planes=planes,
            grid_size=mesh_size,
        )
        
        vtx, faces = mcubes.marching_cubes(grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(), mesh_thres)
        vtx = vtx / (mesh_size - 1) * 2 - 1

        vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=self.device).unsqueeze(0)
        vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
        vtx_colors = (vtx_colors * 255).astype(np.uint8)
        
        mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

        # dump
        os.makedirs(os.path.dirname(dump_mesh_path), exist_ok=True)
        mesh.export(dump_mesh_path)

    def _generate_output_paths(self, image_path: str, omit_prefix: str) -> tuple:
        """Generate standardized output paths for video and mesh files"""
        # Process image path
        image_path = Path(image_path)
        image_stem = image_path.stem
        relative_path = str(Path(image_path.parent).relative_to(omit_prefix) if omit_prefix else image_path.parent)
        # Create output paths with simple names based on source image
        video_path = Path(self.cfg.video_dump) / relative_path / f"{image_stem}.mov"
        mesh_path = Path(self.cfg.mesh_dump) / relative_path / f"{image_stem}.ply"
        render_path = Path(self.cfg.rendered_dump) / relative_path / f"{image_stem}_endo_render.png"
        render_depth_path = Path(self.cfg.rendered_depth) / relative_path / f"{image_stem}_endo_depth.png"
        
        return str(video_path), str(mesh_path), str(render_path), str(render_depth_path)

    def infer_single(self, image_path: str, idx: int, source_cam_dist: float = None, 
                    export_mesh: bool = True):
        """Process single image with both LRM and EndoGaussian components"""
        try:
            total_start = time.time()
            if self.model is None and (self.freeze_endo_gaussian or self.endo_gaussian_trained):
                self.model = self._build_model(self.cfg).to(self.device)
            
            source_size = self.cfg.source_size
            mesh_size = self.cfg.mesh_size
            mesh_thres = self.cfg.mesh_thres
            
            # Get all paths first
            video_path, mesh_path, render_path, render_depth_path = self._generate_output_paths(
                image_path, os.path.dirname(image_path))
            
            # Get EndoGaussian rendered results first (independent of masks)
            t0 = time.time()
            rendered_image, rendered_depth = self._get_endogaussian_render(idx)
            self._save_rendered_outputs(rendered_image, rendered_depth, render_path, render_depth_path)
            endo_time = time.time() - t0
            print(f"EndoGaussian time: {endo_time:.2f}s")
            
            # Process input image and masks
            
            images, masks, num_masks = self._load_and_process_image(image_path, source_size)
            
            # Process each mask separately
            for mask_idx, (image, mask) in enumerate(zip(images, masks)):
                # Only modify mesh path for multiple masks
                current_mesh_path = mesh_path.replace('.ply', f'_mask{mask_idx}.ply') if num_masks > 1 else mesh_path
                
                with torch.no_grad():
                    planes = self.infer_planes(image, source_cam_dist or self.cfg.source_cam_dist)
                    
                    if export_mesh:
                        t0 = time.time()
                        self.infer_mesh(planes, mesh_size, mesh_thres, current_mesh_path)
                        mesh_time = time.time() - t0
                        print(f"Mesh time: {mesh_time:.2f}s")
        
            total_time = time.time() - total_start
            logger.info(f"Total processing time: {total_time:.4f}s\n")
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            raise

    def _load_and_process_image(self, image_path: str, target_size: int):
        """Load and preprocess image and mask(s)"""
        # Load image and mask
        if image_path.find('images') == -1:
            mask_path = image_path.replace('left_finalpass', 'Annotations')
        else:
            mask_path = image_path.replace('images', 'Annotations')

        image = torch.from_numpy(np.array(Image.open(image_path))).float().to(self.device)
        mask_rgb = torch.from_numpy(np.array(Image.open(mask_path))).float().to(self.device)
        
        # Process RGB mask to separate binary masks
        # Convert to (H,W,3) if not already
        if len(mask_rgb.shape) == 2:
            mask_rgb = mask_rgb.unsqueeze(-1).repeat(1, 1, 3)
        
        # Find unique colors in mask (excluding black background)
        mask_colors = torch.unique(mask_rgb.reshape(-1, 3), dim=0)
        mask_colors = mask_colors[mask_colors.sum(dim=1) > 0]  # Remove black background
        
        processed_images = []
        processed_masks = []
        
        # Process image for each mask
        for color in mask_colors:
            # Create binary mask for this color
            color_mask = torch.all(mask_rgb == color.unsqueeze(0).unsqueeze(0), dim=-1)
            color_mask = color_mask.float()
            
            # Normalize and prepare mask
            mask = color_mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            mask = torch.clamp(mask, 0, 1)
            
            # Process image
            image_normalized = image.permute(2, 0, 1).unsqueeze(0) / 255.0
            masked_image = image_normalized * mask + (1 - mask)
            resized_image = torch.nn.functional.interpolate(
                masked_image, 
                size=(target_size, target_size),
                mode='bicubic', 
                align_corners=True
            )
            processed_images.append(torch.clamp(resized_image, 0, 1))
            processed_masks.append(mask)
            
        return processed_images, processed_masks, len(mask_colors)

    def _save_rendered_outputs(self, rendered_image, rendered_depth, render_path, render_depth_path):
        """Save EndoGaussian rendered outputs"""
        os.makedirs(os.path.dirname(render_path), exist_ok=True)
        
        os.makedirs(os.path.dirname(render_depth_path), exist_ok=True)
        
        try:
            # Detach tensors before saving
            rendered_image = rendered_image.detach()
            rendered_depth = rendered_depth.detach()

            torchvision.utils.save_image(rendered_image, render_path)
            # rendered_image_np = rendered_image.cpu().numpy()
            # rendered_image_np = rendered_image_np.transpose(1, 2, 0)
            # rendered_image_np = cv2.cvtColor(rendered_image_np, cv2.COLOR_RGB2BGR)
            # rendered_image_np = (rendered_image_np * 255).astype(np.uint8)
            # cv2.imwrite(render_path, rendered_image_np)
            # torchvision.utils.save_image(rendered_depth, render_depth_path)

            rendered_depth_np = rendered_depth.cpu().numpy().squeeze()
            depth_uint8 = rendered_depth_np.astype(np.uint8)
            cv2.imwrite(render_depth_path, depth_uint8)
        except Exception as e:
            logger.error(f"Failed to save rendered outputs: {str(e)}")

    def infer(self):
        try:
            image_paths = []
            input_path = Path(self.cfg.image_input)
            
            if input_path.is_file():
                omit_prefix = str(input_path.parent)
                image_paths.append(str(input_path))
            else:
                omit_prefix = str(input_path)
                image_paths.extend([
                    str(p) for p in input_path.rglob("*.png")
                ])
                image_paths.sort()

            # Distribute work across DDP workers
            image_paths = image_paths[self.accelerator.process_index::self.accelerator.num_processes]
            for idx, image_path in enumerate(tqdm(image_paths, disable=not self.accelerator.is_local_main_process)):
                try:
                    video_path, mesh_path, _, _ = self._generate_output_paths(image_path, omit_prefix)
                    
                    self.infer_single(
                        image_path,
                        idx,  # Pass the enumerated index
                        source_cam_dist=None,
                        # export_video=self.cfg.export_video,
                        export_mesh=self.cfg.export_mesh,
                        # dump_video_path=video_path,
                        # dump_mesh_path=mesh_path
                    )
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def _build_model(self, cfg):
        """Build and return the LRM model"""
        from openlrm.models import model_dict
        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        model = hf_model_cls.from_pretrained(cfg.model_name)
        return model
