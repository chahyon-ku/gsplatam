from dataclasses import dataclass
from gsplat import rasterization, rasterization_2dgs
import nvtx
import torch
import torch.nn.functional as F

from gsplatam.geometry import build_transform


@dataclass
class Camera:
    viewmats: torch.Tensor
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
    viewmat = build_transform(
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
        'viewmats': viewmat[None],
    }
    return rendervar


class Renderer:
    def __init__(self, camera: Camera):
        self.camera = camera

    @nvtx.annotate("Renderer.__call__")
    def __call__(
        self,
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
    ):
        renders, alphas, info = rasterization(
            means=means,  # [N, 3]
            quats=quats,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacities,  # [N,]
            colors=colors,
            render_mode='RGB+ED',
            viewmats=viewmats,  # [C, 4, 4]
            Ks=self.camera.Ks,  # [C, 3, 3]
            width=self.camera.width,
            height=self.camera.height,
            near_plane=self.camera.near_plane,
            far_plane=self.camera.far_plane,
            eps2d=0,
            packed=True,
            sh_degree=None,
        )
        # [C, H, W, 3] -> [3, H, W]
        renders = renders[0].permute(2, 0, 1)
        alphas = alphas[0].permute(2, 0, 1)

        return renders[:-1], renders[-1:], alphas