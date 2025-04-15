import os
import time
from typing import Tuple
from gsplat.rendering import rasterization
import hydra
import nvtx
import torch
import torch.nn.functional as F
import numpy as np
import viser
import nerfview
from torch.utils.data import DataLoader
from tqdm import tqdm

from SplaTAM.utils.common_utils import save_params

from gsplatam.eval import eval
from gsplatam.geometry import build_transform, get_pointcloud
from gsplatam.gs import get_loss, get_non_presence_mask, initialize_new_params, initialize_optimizer, initialize_params, prune_gaussians, keyframe_selection_overlap
from gsplatam.renderer import Camera


def initialize_first_timestep(
    # dataset,
    color,      # (C, H, W)
    depth,      # (C, H, W)
    intrinsics, # (3, 3)
    w2c,       # (4, 4)
    #
    num_frames,
    scene_radius_depth_ratio, 
    densify_factor,
    mean_sq_dist_method,
    gaussian_distribution,
):

    # Setup Camera
    cam = Camera(intrinsics[None], color.shape[2], color.shape[1])

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    color = color[:, ::densify_factor, ::densify_factor]
    depth = depth[:, ::densify_factor, ::densify_factor]
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, intrinsics / densify_factor, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    params = initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution)

    variables = {'scene_radius': torch.max(depth) / scene_radius_depth_ratio}

    return params, variables, w2c, cam


@torch.no_grad()
def initialize_camera_pose(params, time_idx, forward_prop):
    if time_idx == 0:
        # already init
        pass
    elif time_idx > 1 and forward_prop:
        # Initialize the camera pose for the current frame based on a constant velocity model
        # Rotation
        prev_rot1 = F.normalize(params['cam_unnorm_rots'][time_idx-1].detach(), dim=0)
        prev_rot2 = F.normalize(params['cam_unnorm_rots'][time_idx-2].detach(), dim=0)
        new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2), dim=0)
        params['cam_unnorm_rots'][time_idx] = new_rot.detach()
        # Translation
        prev_tran1 = params['cam_trans'][time_idx-1].detach()
        prev_tran2 = params['cam_trans'][time_idx-2].detach()
        new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
        params['cam_trans'][time_idx] = new_tran.detach()

        # params['cam_unnorm_rots'][time_idx] = params['cam_unnorm_rots'][time_idx-1].detach()
        # params['cam_trans'][time_idx] = params['cam_trans'][time_idx-1].detach()
    else:
        # Initialize the camera pose for the current frame
        params['cam_unnorm_rots'][time_idx] = params['cam_unnorm_rots'][time_idx-1].detach()
        params['cam_trans'][time_idx] = params['cam_trans'][time_idx-1].detach()


def track_frame(
    config,
    render_fn,
    params,
    optimizer,
    time_idx,
    color,
    depth,
    cam,
    first_frame_w2c,
    # gt_w2c_all_frames,
):
    tracking_cam = Camera(
        cam.Ks * color.shape[1] / cam.height,
        color.shape[2],
        color.shape[1]
    )
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    tracking_start_time = time.time()
    loss = torch.zeros(1).to('cpu')
    if time_idx > 0 and not config['tracking']['use_gt_poses']:
        curr_data = {
            'cam': tracking_cam,
            'im': color,
            'depth': depth,
            'id': time_idx,
            'w2c': first_frame_w2c,
            # 'iter_gt_w2c_list': gt_w2c_all_frames
        }

        # Reset Optimizer & Learning Rates for tracking
        # Keep Track of Best Candidate Rotation & Translation
        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][[time_idx]].detach().clone()
        candidate_cam_tran = params['cam_trans'][[time_idx]].detach().clone()
        current_min_loss = float(1e20)
        # Tracking Optimization
        iter = 0
        do_continue_slam = False
        num_iters_tracking = config['tracking']['num_iters']
        while True:
            iter_start_time = time.time()
            # Loss for current frame
            loss, losses = get_loss(
                render_fn,
                params,
                curr_data,
                time_idx,
                config['tracking']['loss_weights'],
                config['tracking']['use_sil_for_loss'],
                config['tracking']['sil_thres'],
                config['tracking']['use_l1'],
                config['tracking']['ignore_outlier_depth_loss'],
                tracking=True
            )
            # Backprop
            loss.backward()
            # Optimizer Update
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                # Save the best candidate rotation & translation
                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_cam_unnorm_rot = params['cam_unnorm_rots'][[time_idx]].detach().clone()
                    candidate_cam_tran = params['cam_trans'][[time_idx]].detach().clone()
                    # candidate_cam_unnorm_rot = params['cam_unnorm_rot'].detach().clone()
                    # candidate_cam_tran = params['cam_tran'].detach().clone()
            # Update the runtime numbers
            iter_end_time = time.time()
            tracking_iter_time_sum += iter_end_time - iter_start_time
            tracking_iter_time_count += 1
            # Check if we should stop tracking
            iter += 1
            if iter == num_iters_tracking:
                if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:
                    break
                elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam:
                    do_continue_slam = True
                    num_iters_tracking = 2*num_iters_tracking
                else:
                    break

        # Copy over the best candidate rotation & translation
        with torch.no_grad():
            params['cam_unnorm_rots'][[time_idx]] = candidate_cam_unnorm_rot
            params['cam_trans'][[time_idx]] = candidate_cam_tran
    tracking_end_time = time.time()
    return tracking_iter_time_sum, tracking_iter_time_count, tracking_end_time - tracking_start_time, 1, loss.cpu()


@torch.no_grad()
def densify_frame(
    config,
    render_fn,
    params,
    time_idx,
    color,
    depth,
    cam,
    first_frame_w2c,
    # gt_w2c_all_frames,
    # sil_thres, 
    # mean_sq_dist_method,
    # gaussian_distribution
):
    densify_cam = Camera(
        cam.Ks * color.shape[1] / cam.height,
        color.shape[2],
        color.shape[1]
    )

    # RGB, Depth, and Silhouette Rendering
    curr_data = {
        'cam': densify_cam,
        'im': color,
        'depth': depth,
        'id': time_idx,
        'w2c': first_frame_w2c,
        # 'iter_gt_w2c_list': gt_w2c_all_frames
    }
    im, depth, silhouette = render_fn(curr_data['cam'], params, time_idx, False, False)
    
    non_presence_mask = get_non_presence_mask(curr_data['depth'][0], depth, silhouette, config['mapping']['sil_thres'])

    # Get the new frame Gaussians based on the Silhouette
    if torch.any(non_presence_mask):
        # Get the new pointcloud in the world frame
        curr_w2c = build_transform(
            params['cam_trans'][time_idx].detach(),
            params['cam_unnorm_rots'][time_idx].detach()
        )
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['cam'].Ks[0],
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=config['mean_sq_dist_method'])
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, config['gaussian_distribution'])
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))

    return params


def map_frame(
    config,
    render_fn,
    params,
    variables,
    mapping_optimizer,
    time_idx,
    color,
    depth,
    cam,
    first_frame_w2c,
    # gt_w2c_all_frames,
    keyframe_list
):
    with nvtx.annotate(f'keyframe selection'):
        with torch.no_grad():
            mapping_cam = Camera(
                cam.Ks * color.shape[1] / cam.height,
                color.shape[2],
                color.shape[1]
            )
            # Get the current estimated rotation & translation
            curr_w2c = build_transform(
                params['cam_trans'][time_idx].detach(),
                params['cam_unnorm_rots'][time_idx].detach()
            )
            # Select Keyframes for Mapping
            num_keyframes = config['mapping_window_size']-2
            selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, mapping_cam.Ks[0], keyframe_list[:-1], num_keyframes)
            selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
            if len(keyframe_list) > 0:
                # Add last keyframe to the selected keyframes
                selected_time_idx.append(keyframe_list[-1]['id'])
                selected_keyframes.append(len(keyframe_list)-1)
            # Add current frame to the selected keyframes
            selected_time_idx.append(time_idx)
            selected_keyframes.append(-1)

    # Mapping
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    mapping_start_time = time.time()
    loss = torch.zeros(1).to('cpu')
    for iter in range(config['mapping']['num_iters']):
        iter_start_time = time.time()
        # Randomly select a frame until current time step amongst keyframes
        rand_idx = np.random.randint(0, len(selected_keyframes))
        selected_rand_keyframe_idx = selected_keyframes[rand_idx]
        if selected_rand_keyframe_idx == -1:
            # Use Current Frame Data
            iter_time_idx = time_idx
            iter_color = color
            iter_depth = depth
        else:
            # Use Keyframe Data
            iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
            iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
            iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
        # iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
        mapping_cam = Camera(
            cam.Ks * iter_color.shape[1] / cam.height,
            iter_color.shape[2],
            iter_color.shape[1]
        )
        iter_data = {
            'cam': mapping_cam,
            'im': iter_color,
            'depth': iter_depth,
            'id': iter_time_idx,# 'intrinsics': intrinsics,
            'w2c': first_frame_w2c,
            # 'iter_gt_w2c_list': iter_gt_w2c
        }
        # Loss for current frame
        loss, losses = get_loss(
            render_fn,
            params,
            iter_data,
            iter_time_idx,
            config['mapping']['loss_weights'],
            config['mapping']['use_sil_for_loss'],
            config['mapping']['sil_thres'],
            config['mapping']['use_l1'],
            config['mapping']['ignore_outlier_depth_loss'],
            mapping=True,
            do_ba=True,
        )
        # Backprop
        with nvtx.annotate(f'backprop'):
            loss.backward()
        with torch.no_grad():
            # Prune Gaussians
            if config['mapping']['prune_gaussians']:
                prune_gaussians(params, variables, mapping_optimizer, iter, config['mapping']['pruning_dict'])
            # Optimizer Update
            with nvtx.annotate(f'optimizer step'):
                mapping_optimizer.step()
            mapping_optimizer.zero_grad(set_to_none=True)
        # Update the runtime numbers
        iter_end_time = time.time()
        mapping_iter_time_sum += iter_end_time - iter_start_time
        mapping_iter_time_count += 1
    # Update the runtime numbers
    mapping_end_time = time.time()

    return mapping_iter_time_sum, mapping_iter_time_count, mapping_end_time - mapping_start_time, 1, loss.cpu()


def slam_frame(
    config,
    render_fn,
    params,
    variables,
    #
    time_idx,
    color,
    depth,
    cam,
    keyframe_list,
    #
    first_frame_w2c,
    # gt_w2c_all_frames,
):
    tracking_metrics, mapping_metrics = None, None
    # Tracking
    with nvtx.annotate(f'tracking {time_idx}'):
        # Initialize the camera pose for the current frame
        initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])
        tracking_optimizer = initialize_optimizer(
            params, # {k: v for k, v in params.items() if k in ['cam_unnorm_rots', 'cam_trans']},
            config['tracking']['lrs']
        )
        tracking_metrics = track_frame(
            config,
            render_fn,
            params,
            tracking_optimizer,
            time_idx,
            color[:, ::config['tracking_factor'], ::config['tracking_factor']],
            depth[:, ::config['tracking_factor'], ::config['tracking_factor']],
            cam,
            first_frame_w2c,
            # gt_w2c_all_frames,
        )
    
    # Densification & KeyFrame-based Mapping
    if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
        # Densification
        if config['mapping']['add_new_gaussians'] and time_idx > 0:
            with nvtx.annotate(f'densification {time_idx}'):
                densify_frame(
                    config,
                    render_fn,
                    params,
                    time_idx,
                    color[:, ::config['densify_factor'], ::config['densify_factor']],
                    depth[:, ::config['densify_factor'], ::config['densify_factor']],
                    cam,
                    first_frame_w2c,
                    # gt_w2c_all_frames,
                )
        
        with nvtx.annotate(f'mapping {time_idx}'):
            mapping_optimizer = initialize_optimizer(params, config['mapping']['lrs'])
            mapping_metrics = map_frame(
                config,
                render_fn,
                params,
                variables,
                mapping_optimizer,
                time_idx,
                color[:, ::config['mapping_factor'], ::config['mapping_factor']],
                depth[:, ::config['mapping_factor'], ::config['mapping_factor']],
                cam,
                first_frame_w2c,
                # gt_w2c_all_frames,
                keyframe_list
            )
    
    # Add frame to keyframe list
    if (
        (time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0)
    ):# and (not torch.isinf(gt_w2c_all_frames[-1]).any()) and (not torch.isnan(gt_w2c_all_frames[-1]).any()):
        with torch.no_grad():
            # Get the current estimated rotation & translation
            curr_w2c = build_transform(
                params['cam_trans'][time_idx].detach(),
                params['cam_unnorm_rots'][time_idx].detach()
            )
            # Initialize Keyframe Info
            curr_keyframe = {
                'id': time_idx,
                'est_w2c': curr_w2c,
                'color': color[:, ::config['mapping_factor'], ::config['mapping_factor']],
                'depth': depth[:, ::config['mapping_factor'], ::config['mapping_factor']]
            }
            # Add to keyframe list
            keyframe_list.append(curr_keyframe)
    
    return tracking_metrics, mapping_metrics


def rgbd_slam(config: dict):
    render_fn = hydra.utils.instantiate(config['render_fn'])

    # Create Output Directories
    output_dir = os.path.join(config['workdir'], config['run_name'])
    eval_dir = os.path.join(output_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    # Get Device
    device = torch.device(config['primary_device'])

    # Load Dataset
    print('Loading Dataset ...')
    dataset = hydra.utils.instantiate(config['data']['dataset'],)
    num_frames = config['data']['num_frames']
    if num_frames == -1:
        num_frames = len(dataset)

    # Initialize Parameters & Canoncial Camera parameters
    color, depth, intrinsics, pose = dataset[0]
    params, variables, first_frame_w2c, cam = initialize_first_timestep(
        # dataset,
        color.permute(2, 0, 1) / 255, # (H, W, C) -> (C, H, W)
        depth.permute(2, 0, 1), # (H, W, C) -> (C, H, W)
        intrinsics[:3, :3],
        torch.linalg.inv(pose),
        num_frames, 
        config['scene_radius_depth_ratio'],
        config['densify_factor'],
        config['mean_sq_dist_method'],
        config['gaussian_distribution'],
    )
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0
    checkpoint_time_idx = 0
    
    dataset.device = 'cpu'
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    dataloader_iter = dataloader.__iter__()
    tracking_loss = torch.zeros(1).to('cpu')
    mapping_loss = torch.zeros(1).to('cpu')

    if config.viewer:
        @torch.no_grad()
        def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
            """Callable function for the viewer."""
            W, H = img_wh
            c2w = camera_state.c2w
            K = camera_state.get_K(img_wh)
            c2w = torch.from_numpy(c2w).float().to('cuda')
            K = torch.from_numpy(K).float().to('cuda')
            
            if params['log_scales'].shape[1] == 1:
                log_scales = torch.tile(params['log_scales'], (1, 3))
            else:
                log_scales = params['log_scales']
            rendervar = {
                'means': params['means3D'],
                'quats': F.normalize(params['unnorm_rotations']),
                'scales': torch.exp(log_scales),
                'opacities': torch.sigmoid(params['logit_opacities'][:, 0]),
                'colors': params['rgb_colors'],
                'viewmats': torch.linalg.inv(c2w)[None],
            }
            renders, silhouette, info = rasterization(
                **rendervar,
                render_mode='RGB',
                Ks=K[None],  # [C, 3, 3]
                width=W,
                height=H,
                eps2d=0,
                packed=True,
                sh_degree=None,
            )
            return renders[0].cpu().numpy()
        
        server = viser.ViserServer(port=8080, verbose=False)
        viewer = nerfview.Viewer(
            server=server,
            render_fn=viewer_render_fn,
            mode="training",
        )

    # Iterate over Scan
    time_idx_tqdm = tqdm(list(range(checkpoint_time_idx, num_frames)))
    for time_idx in time_idx_tqdm:
        if config.viewer:
            viewer.lock.acquire()
            tic = time.time()
        # Load RGBD frames incrementally instead of all frames
        with nvtx.annotate('dataset[time_idx]'):
            color, depth, _, gt_pose = next(dataloader_iter)
            color = color[0].to(device)
            depth = depth[0].to(device)
            gt_pose = gt_pose[0].to(device)
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        gt_w2c_all_frames.append(gt_w2c)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
            
        time_idx_tqdm.set_postfix_str(
            f'num_gaussians: {params['means3D'].shape[0]}'
            f' | tracking_loss: {tracking_loss.item():.4f}'
            f' | mapping_loss: {mapping_loss.item():.4f}'
        )

        tracking_metrics, mapping_metrics = slam_frame(
            config,
            render_fn,
            params,
            variables,
            #
            time_idx,
            color,
            depth,
            cam,
            keyframe_list,
            #
            first_frame_w2c,
        )

        if tracking_metrics is not None:
            tracking_iter_time_sum += tracking_metrics[0]
            tracking_iter_time_count += tracking_metrics[1]
            tracking_frame_time_sum += tracking_metrics[2]
            tracking_frame_time_count += tracking_metrics[3]
            tracking_loss = tracking_metrics[4].cpu()
        
        if mapping_metrics is not None:
            mapping_iter_time_sum += mapping_metrics[0]
            mapping_iter_time_count += mapping_metrics[1]
            mapping_frame_time_sum += mapping_metrics[2]
            mapping_frame_time_count += mapping_metrics[3]
            mapping_loss = mapping_metrics[4].cpu()

        if config.viewer:
            viewer.lock.release()
            num_train_rays_per_step = cam.height * cam.width
            num_train_steps_per_sec = 1.0 / (time.time() - tic)
            num_train_rays_per_sec = (
                num_train_rays_per_step * num_train_steps_per_sec
            )
            # Update the viewer state.
            viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            viewer.update(time_idx, num_train_rays_per_step)

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f'\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms')
    print(f'Average Tracking/Frame Time: {tracking_frame_time_avg} s')
    print(f'Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms')
    print(f'Average Mapping/Frame Time: {mapping_frame_time_avg} s')

    # Evaluate Final Parameters
    eval(
        render_fn,
        dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
        mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
        eval_every=config['eval_every']
    )

    # Add Camera Parameters to Save them
    params['timestep'] = time_idx
    params['intrinsics'] = cam.Ks[0].detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = config['data']['desired_image_width']
    params['org_height'] = config['data']['desired_image_height']
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)

