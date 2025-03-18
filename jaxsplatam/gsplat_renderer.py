from dataclasses import dataclass
from gsplat import rasterization
import torch


@dataclass
class Camera:
    viewmats: torch.Tensor
    Ks: torch.Tensor
    width: int
    height: int
    near_plane: float
    far_plane: float


def setup_camera(width, height, K, viewmat, near_plane=0.01, far_plane=100) -> Camera:
    cam = Camera(
        viewmats=torch.tensor(viewmat[None]).cuda().float(),
        Ks=torch.tensor(K[None]).cuda().float(),
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
    )
    return cam


class GsplatRenderer:
    def __init__(self, camera: Camera):
        self.camera = camera

    def __call__(
        self,
        means3D,
        means2D,
        opacities,
        shs = None,
        colors_precomp = None,
        scales = None,
        rotations = None,
        cov3D_precomp = None,
    ):
        if colors_precomp is None:
            raise Exception('Please provide precomputed colors!')
        
        if scales is None or rotations is None:
            raise Exception('Please provide scale and rotation!')

        render_colors, render_alphas, info = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacities.squeeze(-1),  # [N,]
            colors=colors_precomp,
            render_mode='RGB+ED',
            viewmats=self.camera.viewmats,  # [1, 4, 4]
            Ks=self.camera.Ks,  # [1, 3, 3]
            width=self.camera.width,
            height=self.camera.height,
            near_plane=self.camera.near_plane,
            far_plane=self.camera.far_plane,
            eps2d=0,
            packed=False,
            sh_degree=None,
        )
        # [1, H, W, 3] -> [3, H, W]
        render_colors = render_colors[0].permute(2, 0, 1)
        render_alphas = render_alphas[0].permute(2, 0, 1)
        radii = info["radii"].squeeze(0) # [N,]
        try:
            info["means2d"].retain_grad() # [1, N, 2]
        except:
            pass

        return render_colors[:-1], radii, render_alphas, render_colors[-1:], info["means2d"]