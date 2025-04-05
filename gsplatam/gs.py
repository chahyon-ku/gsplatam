import nvtx
import torch
from gsplat import fully_fused_projection

from SplaTAM.utils.keyframe_selection import get_pointcloud
from SplaTAM.utils.slam_external import inverse_sigmoid, update_params_and_optimizer


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
            # torch.cuda.empty_cache()
        
        # Reset Opacities for all Gaussians
        if iter > 0 and iter % prune_dict['reset_opacities_every'] == 0 and prune_dict['reset_opacities']:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)
    
    return params


@torch.compile
def get_percent_inside(pts, est_w2c, intrinsics, width, height):
    N = pts.shape[0]
    C = est_w2c.shape[0]
    covars = torch.empty((N, 6), device=pts.device, dtype=pts.dtype)
    camera_ids, *_ = fully_fused_projection(
        means=pts,
        covars=covars,
        quats=None,
        scales=None,
        viewmats=est_w2c,
        Ks=intrinsics,
        width=width,
        height=height,
        packed=True,
    )
    percent_inside = torch.bincount(camera_ids, minlength=C) / N
    return percent_inside


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
        pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

        viewmats = torch.stack([keyframe['est_w2c'] for keyframe in keyframe_list], dim=0)
        Ks = intrinsics[None].repeat(viewmats.shape[0], 1, 1)
        percent_insides = get_percent_inside(pts, viewmats, Ks, width, height)
        keyframe_list = torch.arange(len(keyframe_list), device=pts.device)
        keyframe_list = keyframe_list[percent_insides > 0]
        keyframe_list = keyframe_list[torch.randperm(keyframe_list.shape[0], device=pts.device)][:k]
        keyframe_list = keyframe_list.cpu().numpy().tolist()

        return keyframe_list