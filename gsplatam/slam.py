import os
import time
import hydra
import nvtx
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from SplaTAM.utils.common_utils import save_params_ckpt, save_params
from SplaTAM.utils.slam_helpers import matrix_to_quaternion

from gsplatam.eval import eval
from gsplatam.geometry import build_transform, get_pointcloud
from gsplatam.gs import get_loss, get_non_presence_mask, initialize_new_params, initialize_optimizer, initialize_params, prune_gaussians, keyframe_selection_overlap
from gsplatam.renderer import Camera


def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, 
                              mean_sq_dist_method, densify_factor, gaussian_distribution):
    # Get RGB-D Data & Camera Parameters
    color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

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
    
    return params


def track_frame(
    config,
    render_fn,
    params,
    optimizer,
    time_idx,
    color,
    depth,
    tracking_cam,
    first_frame_w2c,
    gt_w2c_all_frames,
):
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    tracking_start_time = time.time()
    loss = torch.zeros(1).to('cpu')
    if time_idx > 0 and not config['tracking']['use_gt_poses']:
        curr_data = {
            'cam': tracking_cam,
            'im': color[:, ::config['tracking_factor'], ::config['tracking_factor']],
            'depth': depth[:, ::config['tracking_factor'], ::config['tracking_factor']],
            'id': time_idx,
            'w2c': first_frame_w2c,
            'iter_gt_w2c_list': gt_w2c_all_frames
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
    elif time_idx > 0 and config['tracking']['use_gt_poses']:
        with torch.no_grad():
            # Get the ground truth pose relative to frame 0
            rel_w2c = gt_w2c_all_frames[-1]
            rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
            rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
            rel_w2c_tran = rel_w2c[:3, 3].detach()
            # Update the camera parameters
            params['cam_unnorm_rots'][[time_idx]] = rel_w2c_rot_quat
            params['cam_trans'][[time_idx]] = rel_w2c_tran
    # Update the runtime numbers
    tracking_end_time = time.time()
    return tracking_iter_time_sum, tracking_iter_time_count, tracking_end_time - tracking_start_time, 1, loss


@torch.no_grad()
def densify_frame(
    config,
    render_fn,
    params, 
    optimizer,
    time_idx,
    color,
    depth,
    densify_cam,
    first_frame_w2c,
    gt_w2c_all_frames,
    # sil_thres, 
    # mean_sq_dist_method,
    # gaussian_distribution
):
    # RGB, Depth, and Silhouette Rendering
    curr_data = {
        'cam': densify_cam,
        'im': color[:, ::config['densify_factor'], ::config['densify_factor']],
        'depth': depth[:, ::config['densify_factor'], ::config['densify_factor']],
        'id': time_idx,
        'w2c': first_frame_w2c,
        'iter_gt_w2c_list': gt_w2c_all_frames
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

    return params, optimizer


def map_frame(
    config,
    render_fn,
    params,
    variables,
    mapping_optimizer,
    time_idx,
    color,
    depth,
    mapping_cam,
    first_frame_w2c,
    gt_w2c_all_frames,
    keyframe_list
):
    with nvtx.annotate(f'keyframe selection'):
        with torch.no_grad():
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
        iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1]
        iter_data = {
            'cam': mapping_cam,
            'im': iter_color,
            'depth': iter_depth,
            'id': iter_time_idx,# 'intrinsics': intrinsics,
            'w2c': first_frame_w2c,
            'iter_gt_w2c_list': iter_gt_w2c
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
                params = prune_gaussians(params, variables, mapping_optimizer, iter, config['mapping']['pruning_dict'])
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

    return mapping_iter_time_sum, mapping_iter_time_count, mapping_end_time - mapping_start_time, 1, loss


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
    params, variables, first_frame_w2c, cam = initialize_first_timestep(
        dataset,
        num_frames, 
        config['scene_radius_depth_ratio'],
        config['mean_sq_dist_method'],
        config['densify_factor'],
        config['gaussian_distribution']
    )
    tracking_cam, densify_cam, mapping_cam = [Camera(
        cam.Ks / factor,
        cam.width // factor,
        cam.height // factor,
    ) for factor in [
        config['tracking_factor'],
        config['densify_factor'],
        config['mapping_factor']
    ]]
    
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
    prev_tracking_loss = torch.zeros(1).to('cpu')
    prev_mapping_loss = torch.zeros(1).to('cpu')

    # Iterate over Scan
    mapping_optimizer = initialize_optimizer(params, config['mapping']['lrs'])
    time_idx_tqdm = tqdm(list(range(checkpoint_time_idx, num_frames)))
    for time_idx in time_idx_tqdm:
        # Load RGBD frames incrementally instead of all frames
        with nvtx.annotate('dataset[time_idx]'):
            color, depth, _, gt_pose = next(dataloader_iter)
            color = color[0].to(device)
            depth = depth[0].to(device)
            gt_pose = gt_pose[0].to(device)
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)

        time_idx_tqdm.set_postfix_str(
            f'num_gaussians: {params["means3D"].shape[0]}'
            f' | tracking_loss: {prev_tracking_loss.item():.4f}'
            f' | mapping_loss: {prev_mapping_loss.item():.4f}'
        )

        # Tracking
        with nvtx.annotate(f'tracking {time_idx}'):
            # Initialize the camera pose for the current frame
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop'])
            tracking_optimizer = initialize_optimizer(
                params, # {k: v for k, v in params.items() if k in ['cam_unnorm_rots', 'cam_trans']},
                config['tracking']['lrs']
            )
            metrics = track_frame(
                config,
                render_fn,
                params,
                tracking_optimizer,
                time_idx,
                color,
                depth,
                tracking_cam,
                first_frame_w2c,
                gt_w2c_all_frames,
            )
            tracking_iter_time_sum += metrics[0]
            tracking_iter_time_count += metrics[1]
            tracking_frame_time_sum += metrics[2]
            tracking_frame_time_count += metrics[3]
            prev_tracking_loss = metrics[4].cpu()
        
        # Densification & KeyFrame-based Mapping
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                with nvtx.annotate(f'densification {time_idx}'):
                    params, mapping_optimizer = densify_frame(
                        config,
                        render_fn,
                        params,
                        mapping_optimizer,
                        time_idx,
                        color,
                        depth,
                        densify_cam,
                        first_frame_w2c,
                        gt_w2c_all_frames,
                    )
            
            with nvtx.annotate(f'mapping {time_idx}'):
                mapping_optimizer = initialize_optimizer(params, config['mapping']['lrs'])
                metrics = map_frame(
                    config,
                    render_fn,
                    params,
                    variables,
                    mapping_optimizer,
                    time_idx,
                    color,
                    depth,
                    mapping_cam,
                    first_frame_w2c,
                    gt_w2c_all_frames,
                    keyframe_list
                )
                mapping_iter_time_sum += metrics[0]
                mapping_iter_time_count += metrics[1]
                mapping_frame_time_sum += metrics[2]
                mapping_frame_time_count += metrics[3]
                prev_mapping_loss = metrics[4].cpu()
        
        # Add frame to keyframe list
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(gt_w2c_all_frames[-1]).any()) and (not torch.isnan(gt_w2c_all_frames[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_w2c = build_transform(
                    params['cam_trans'][time_idx].detach(),
                    params['cam_unnorm_rots'][time_idx].detach()
                )
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config['checkpoint_interval'] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config['workdir'], config['run_name'])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f'keyframe_time_indices{time_idx}.npy'), np.array(keyframe_time_indices))

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
