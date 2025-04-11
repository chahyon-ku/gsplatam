from dataclasses import dataclass
from typing import Dict, Tuple
import nvtx
import torch
import torch.nn.functional as F

from gsplat import rasterization, rasterization_2dgs
from gsplatam.geometry import build_transform
from SplaTAM.utils.gs_helpers import transform_to_frame, transformed_params2depthplussilhouette, transformed_params2rendervar, Renderer
from SplaTAM.utils.recon_helpers import setup_camera


@dataclass
class Camera:
    Ks: torch.Tensor
    width: int
    height: int
    near_plane: float = 0.01
    far_plane: float = 100


@nvtx.annotate('get_rendervar')
def get_rendervar(
    params, iter_time_idx, gaussians_grad, camera_grad
):
    cam_rot = params['cam_unnorm_rots'][iter_time_idx]
    cam_tran = params['cam_trans'][iter_time_idx]

    viewmats = build_transform(
        cam_tran if camera_grad else cam_tran.detach(),
        cam_rot if camera_grad else cam_rot.detach()
    )

    if params['log_scales'].shape[1] == 1:
        log_scales = torch.tile(params['log_scales'], (1, 3))
    else:
        log_scales = params['log_scales']
    # Initialize Render Variables
    rendervar = {
        'means': params['means3D'] if gaussians_grad else params['means3D'].detach(),
        'quats': F.normalize(params['unnorm_rotations'] if gaussians_grad else params['unnorm_rotations'].detach()),
        'scales': torch.exp(log_scales),
        'opacities': torch.sigmoid(params['logit_opacities'][:, 0]),
        'colors': params['rgb_colors'],
        'viewmats': viewmats,
    }
    return rendervar


def render_gsplat(
    camera: Camera,
    params: Dict[str, torch.Tensor],
    iter_time_idx: int,
    gaussians_grad: bool,
    camera_grad: bool,
):
    rendervar = get_rendervar(
        params=params,
        iter_time_idx=iter_time_idx,
        gaussians_grad=gaussians_grad,
        camera_grad=camera_grad
    )

    C = rendervar['viewmats'].shape[0]
    renders, silhouette, info = rasterization(
        **rendervar,
        render_mode='RGB+ED',
        Ks=torch.tile(camera.Ks, (C, 1, 1)),  # [C, 3, 3]
        width=camera.width,
        height=camera.height,
        near_plane=camera.near_plane,
        far_plane=camera.far_plane,
        eps2d=0,
        packed=True,
        sh_degree=None,
    )

    # [C, H, W, 3]
    renders = renders
    silhouette = silhouette
    im, depth = renders[..., :-1], renders[..., -1:]

    return im, depth, silhouette


def render_gsplat_2dgs(
    camera: Camera,
    params: Dict[str, torch.Tensor],
    iter_time_idx: int,
    gaussians_grad: bool,
    camera_grad: bool,
):
    rendervar = get_rendervar(
        params=params,
        iter_time_idx=iter_time_idx,
        gaussians_grad=gaussians_grad,
        camera_grad=camera_grad
    )

    C = rendervar['viewmats'].shape[0]
    (
        renders,
        silhouette,
        # render_normals,
        # render_normals_from_depth,
        # render_distort,
        # render_median,
        info
    ) = rasterization_2dgs(
        **rendervar,
        render_mode='RGB+ED',
        Ks=torch.tile(camera.Ks, (C, 1, 1)),  # [C, 3, 3]
        width=camera.width,
        height=camera.height,
        near_plane=camera.near_plane,
        far_plane=camera.far_plane,
        eps2d=0,
        packed=True,
        sh_degree=None,
    )

    # [C, H, W, 3]
    renders = renders
    silhouette = silhouette
    im, depth = renders[..., :-1], renders[..., -1:]

    return im, depth, silhouette


def get_render_fn(type: str):
    if type == 'gsplat':
        return render_gsplat
    elif type == 'gsplat_2dgs':
        return render_gsplat_2dgs
    else:
        raise ValueError(f"Unknown renderer type: {type}")
