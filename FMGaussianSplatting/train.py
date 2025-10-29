import numpy as np
import random
import os
import torch
from random import randint
from FMGaussianSplatting.utils.loss_utils import l1_loss, ssim, lpips_loss, TV_loss
from FMGaussianSplatting.gaussian_renderer import render, network_gui
import logging
from FMGaussianSplatting.scene import Scene, GaussianModel
from tqdm import tqdm
from FMGaussianSplatting.utils.image_utils import psnr
from FMGaussianSplatting.arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams

from FMGaussianSplatting.utils.timer import Timer

import lpips
from FMGaussianSplatting.utils.scene_utils import render_training_image
from time import time
to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Remove erroneous .s
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, train_iter, timer):
    first_iter = 0
    gaussians.training_setup(opt)
    if checkpoint:
        print("Loading checkpoint from {}".format(checkpoint))
        (model_params, first_iter) = torch.load(checkpoint)
        print("Done loading checkpoint")
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # trainCameras = scene.getTrainCameras().copy()
    # gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    
    final_iter = train_iter
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")

    first_iter += 1
    lpips_model = lpips.LPIPS(net="alex").cuda()
    video_cams = scene.getVideoCameras()
    
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras()
    
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage="stage")["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()
        if stage == 'coarse':
            idx = 0
        else:
            idx = randint(0, len(viewpoint_stack)-1)
        viewpoint_cams = [viewpoint_stack[idx]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        images = []
        depths = []
        gt_images = []
        gt_depths = []
        masks = []
        
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        
        for viewpoint_cam in viewpoint_cams:

            #TODO ignore border pixels
            if dataset.ray_jitter:
                subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
                # subpixel_offset *= 0.0
            else:
                subpixel_offset = None
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset, stage=stage)
            image, depth, viewspace_point_tensor, visibility_filter, radii = \
                render_pkg["render"], render_pkg["depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = viewpoint_cam.original_image.cuda().float()
            gt_depth = viewpoint_cam.original_depth.cuda().float()
            mask = viewpoint_cam.mask.cuda()
            
            images.append(image.unsqueeze(0))
            depths.append(depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            gt_depths.append(gt_depth.unsqueeze(0))
            masks.append(mask.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)
            
        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        rendered_images = torch.cat(images,0)
        rendered_depths = torch.cat(depths, 0)
        gt_images = torch.cat(gt_images,0)
        gt_depths = torch.cat(gt_depths, 0)
        masks = torch.cat(masks, 0)
        
        Ll1 = l1_loss(rendered_images, gt_images, masks)
        
        if (gt_depths!=0).sum() < 10:
            depth_loss = torch.tensor(0.).cuda()
        elif scene.mode == 'binocular':
            rendered_depths[rendered_depths!=0] = 1 / rendered_depths[rendered_depths!=0]
            gt_depths[gt_depths!=0] = 1 / gt_depths[gt_depths!=0]
            depth_loss = l1_loss(rendered_depths, gt_depths, masks)
        else:
            raise ValueError(f"{scene.mode} is not implemented.")
        
        depth_tvloss = TV_loss(rendered_depths)
        img_tvloss = TV_loss(rendered_images)
        tv_loss = 0.03 * (img_tvloss + depth_tvloss)
        
        loss = Ll1 + depth_loss + tv_loss
        psnr_ = psnr(rendered_images, gt_images, masks).mean().double()        
        
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(2e-2, 2e-2, 2e-2)
            loss += tv_loss
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(rendered_images,gt_images)
            loss += opt.lambda_dssim * (1.0-ssim_loss)
        if opt.lambda_lpips !=0:
            lpipsloss = lpips_loss(rendered_images,gt_images,lpips_model)
            loss += opt.lambda_lpips * lpipsloss
        
        loss.backward()

        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == train_iter:
                progress_bar.close()

            # Log and save
            timer.pause()
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, [pipe, background], stage)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if dataset.render_process:
                if (iteration < 1000 and iteration % 10 == 1) \
                    or (iteration < 3000 and iteration % 50 == 1) \
                        or (iteration < 10000 and iteration %  100 == 1) \
                            or (iteration < 60000 and iteration % 100 ==1):
                    render_training_image(scene, gaussians, video_cams, render, pipe, background, stage, iteration-1,timer.get_elapsed_time())
            timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0:
                    size_threshold = 40 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    print("reset opacity")
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < train_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def validate_args(args):
    """Validate arguments and provide sensible defaults"""
    required_attrs = [
        'sh_degree', 'source_path', 'model_path', 'white_background',
        'iterations', 'coarse_iterations', 'mode'
    ]
    
    for attr in required_attrs:
        if not hasattr(args, attr) or getattr(args, attr) is None:
            raise ValueError(f"Missing required argument: {attr}")
            
    # Set dependent parameters
    if not hasattr(args, 'eval'):
        args.eval = True
    if not hasattr(args, 'render_process'):
        args.render_process = False
        
    return args


def _training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname, args):

    args = validate_args(args)
    
    # Log training configuration using print instead of logger
    print("======== Training Configuration ========")
    print(f"Source path: {args.source_path}")
    print(f"Model path: {args.model_path}")
    print(f"Mode: {args.mode}")
    print(f"Iterations: {args.iterations}")
    print(f"Coarse iterations: {args.coarse_iterations}")
    print("=====================================")

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=args.no_fine)
    timer.start()
    # Run training stages
    try:
        # Coarse stage
        logger.info("Starting coarse reconstruction...")
        scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                           checkpoint_iterations, checkpoint, debug_from,
                           gaussians, scene, "coarse", opt.coarse_iterations, timer)
        
        # Fine stage
        if not args.no_fine:
            logger.info("Starting fine reconstruction...")
            scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                              checkpoint_iterations, checkpoint, debug_from,
                              gaussians, scene, "fine", opt.iterations, timer)
                              
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        timer.pause()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def training(parser, config_path=None):
    # Initialize parameter classes
    torch.cuda.empty_cache()
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    # Create args from parser
    args = parser.parse_args([])  # Empty list since we're not using command line args
    args.save_iterations = [2000, 3000, 4000, 5000, 6000, 9000, 10000, 14000, 20000, 30000, 45000, 60000]
    args.test_iterations = [i*500 for i in range(0,120)]
    args.checkpoint_iterations = []
    args.start_checkpoint = None
    args.debug_from = -1
    args.quiet = False

    # Load config if provided
    if config_path:
        import mmcv
        config = mmcv.Config.fromfile(config_path)
        for k, v in config.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    setattr(args, sub_k, sub_v)

    # Extract parameters
    dataset = lp.extract(args)
    hyper = hp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # Make sure iterations is in save_iterations
    if hasattr(args, 'iterations') and args.iterations not in args.save_iterations:
        args.save_iterations.append(args.iterations)

    # Run the training
    return _training(dataset, hyper, opt, pipe, args.test_iterations, args.save_iterations,
                    args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.extra_mark, args)

