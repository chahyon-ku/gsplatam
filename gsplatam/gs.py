import numpy as np
import nvtx
import torch

from fused_ssim import fused_ssim
from SplaTAM.utils.slam_external import cat_params_to_optimizer, inverse_sigmoid, update_params_and_optimizer
from SplaTAM.utils.slam_helpers import l1_loss_v1
from SplaTAM.utils.keyframe_selection import get_pointcloud as get_keyframe_pointcloud

from gsplatam.geometry import build_transform, get_percent_inside, get_pointcloud
from gsplatam.renderer import Renderer, get_rendervar


def remove_points(to_remove, params, optimizer):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ['cam_unnorm_rots', 'cam_trans']]
    for k in keys:
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]
    return params


@nvtx.annotate("prune_gaussians")
def prune_gaussians(params, variables, optimizer, iter, prune_dict):
    if iter <= prune_dict['stop_after']:
        if (iter >= prune_dict['start_after']) and (iter % prune_dict['prune_every'] == 0):
            if iter == prune_dict['stop_after']:
                remove_threshold = prune_dict['final_removal_opacity_threshold']
            else:
                remove_threshold = prune_dict['removal_opacity_threshold']
            # Remove Gaussians with low opacity
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            # Remove Gaussians that are too big
            if iter >= prune_dict['remove_big_after']:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params = remove_points(to_remove, params, optimizer)
        
        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
    
    return params


def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
    if len(keyframe_list) == 0:
        return []
    
    # Radomly Sample Pixel Indices from valid depth pixels
    width, height = gt_depth.shape[2], gt_depth.shape[1]
    valid_depth_indices = torch.where(gt_depth[0] > 0)
    valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
    indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
    sampled_indices = valid_depth_indices[indices]

    # Back Project the selected pixels to 3D Pointcloud
    with nvtx.annotate('get_keyframe_pointcloud'):
        pts = get_keyframe_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)
        # pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

    viewmats = torch.stack([keyframe['est_w2c'] for keyframe in keyframe_list], dim=0)
    Ks = intrinsics[None].repeat(viewmats.shape[0], 1, 1)
    percent_insides = get_percent_inside(pts, viewmats, Ks, width, height)
    keyframe_list = torch.arange(len(keyframe_list), device=pts.device)
    keyframe_list = keyframe_list[percent_insides > 0]
    keyframe_list = keyframe_list[torch.randperm(keyframe_list.shape[0], device=pts.device)][:k]
    keyframe_list = keyframe_list.cpu().numpy().tolist()

    return keyframe_list


@nvtx.annotate('get_non_presence_mask')
@torch.no_grad()
@torch.compile
def get_non_presence_mask(gt_depth, depth, silhouette, sil_thresh):
    non_presence_sil_mask = (silhouette < sil_thresh)
    # Check for new foreground objects by using GT depth
    depth_error = torch.abs(gt_depth - depth) * (gt_depth > 0)
    non_presence_depth_mask = (depth > gt_depth) * (depth_error > 50*depth_error.median())
    # Determine non-presence mask
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)
    valid_depth_mask = gt_depth > 0
    non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
    return non_presence_mask


@nvtx.annotate('add_new_gaussians')
@torch.no_grad()
def add_new_gaussians(
    params, 
    optimizer,
    curr_data,
    sil_thres, 
    time_idx,
    mean_sq_dist_method,
    gaussian_distribution
):
    # RGB, Depth, and Silhouette Rendering
    rendervar = get_rendervar(params, time_idx, False, False)
    im, depth, silhouette = Renderer(camera=curr_data['cam'])(**rendervar)

    non_presence_mask = get_non_presence_mask(curr_data['depth'][0], depth, silhouette, sil_thres)

    # Get the new frame Gaussians based on the Silhouette
    if torch.any(non_presence_mask):
        # Get the new pointcloud in the world frame
        curr_w2c = build_transform(
            params['cam_trans'][time_idx].detach(),
            params['cam_unnorm_rots'][time_idx].detach()
        )
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['cam'].Ks[0],
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method)
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution)
        # for k, v in new_params.items():
        #     params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        params = cat_params_to_optimizer(new_params, params, optimizer)

    return params, optimizer


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist, gaussian_distribution):
    num_pts = init_pt_cld.shape[0]
    mean3_sq_dist = mean3_sq_dist.cpu()
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1))
    logit_opacities = np.zeros((num_pts, 1))
    if gaussian_distribution == 'isotropic':
        log_scales = np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == 'anisotropic':
        log_scales = np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f'Unknown gaussian_distribution {gaussian_distribution}')
    cam_unnorm_rots = np.tile([1, 0, 0, 0], (num_frames, 1))
    cam_trans = np.zeros((num_frames, 3))

    params = {
        'means3D': init_pt_cld[:, :3],      # [num_pts, 3]
        'rgb_colors': init_pt_cld[:, 3:],   # [num_pts, 3]
        'unnorm_rotations': unnorm_rots,    # [num_pts, 4]
        'logit_opacities': logit_opacities, # [num_pts,]
        'log_scales': log_scales,           # [num_pts, 1]
        'cam_unnorm_rots': cam_unnorm_rots, # [num_frames, 4]
        'cam_trans': cam_trans              # [num_frames, 3]
    }

    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_new_params(new_pt_cld, mean3_sq_dist, gaussian_distribution):
    num_pts = new_pt_cld.shape[0]
    unnorm_rotations = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 4]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device='cuda')
    if gaussian_distribution == 'isotropic':
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1))
    elif gaussian_distribution == 'anisotropic':
        log_scales = torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 3))
    else:
        raise ValueError(f'Unknown gaussian_distribution {gaussian_distribution}')
    params = {
        'means3D': new_pt_cld[:, :3],
        'rgb_colors': new_pt_cld[:, 3:],
        'unnorm_rotations': unnorm_rotations,
        'logit_opacities': logit_opacities,
        'log_scales': log_scales,
    }
    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
    
    return params



# torch.compiler.allow_in_graph(fused_ssim)
@nvtx.annotate('compute_loss')
@torch.compile
def compute_loss(
    im, silhouette, depth, curr_data,
    loss_weights, use_sil_for_loss,
    sil_thres, use_l1, ignore_outlier_depth_loss,
    tracking
):
    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = depth > 0
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & (silhouette > sil_thres)

    losses = {}
    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = (torch.abs(curr_data['depth'] - depth) * mask).sum()
        else:
            losses['depth'] = (torch.abs(curr_data['depth'] - depth) * mask).mean()
    
    # RGB Loss
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = (torch.abs(curr_data['im'] - im) * color_mask).sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - fused_ssim(im[None], curr_data['im'][None]))

    losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(losses.values())
    losses['loss'] = loss
    return loss, losses


@nvtx.annotate('scripts.splatam.get_loss')
def get_loss(params, curr_data, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False):

    rendervar = get_rendervar(params, iter_time_idx, not tracking, tracking or (mapping and do_ba))

    # RGB, Depth, and Silhouette Rendering
    im, depth, silhouette = Renderer(camera=curr_data['cam'])(**rendervar)

    loss, losses = compute_loss(
        im, silhouette, depth, curr_data,
        loss_weights, use_sil_for_loss,
        sil_thres, use_l1, ignore_outlier_depth_loss,
        tracking
    )
    return loss, losses
