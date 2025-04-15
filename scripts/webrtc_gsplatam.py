import asyncio
import time
from typing import Tuple
import aiohttp
import cv2
from gsplat.rendering import rasterization
import hydra
import nerfview
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole
from omegaconf import DictConfig, OmegaConf
import open3d as o3d
import torch
import viser
import torch.nn.functional as F

from gsplatam.renderer import Camera
from gsplatam.slam import initialize_first_timestep, slam_frame

OmegaConf.register_new_resolver("eval", eval, replace=True)


class SignalingServer:
    def __init__(self, server_url):
        self.server_url = server_url

    async def retrieve_offer(self):
        url = self.server_url + '/getOffer'
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(url) as resp:
                        return await resp.json()
                except Exception as e:
                    print('Error while requesting an offer:', e)
                    time.sleep(1)

    async def send_answer(self, answer):
        url = self.server_url + '/answer'
        json_answer = {
            'type': 'answer',
            'data': answer
        }
        async with aiohttp.ClientSession() as session:
            try:
                await session.post(url, json=json_answer)
            except Exception as e:
                print('Error while sending the answer:', e)

async def start_receiving_stream(config, server_url):
    signaling_server = SignalingServer(server_url)
    peer_connection = RTCPeerConnection()

    @peer_connection.on("icecandidate")
    async def on_ice_candidate(event):
        if event.candidate is None:
            print("Finished ICE candidate gathering, sending answer.")
            await signaling_server.send_answer(peer_connection.localDescription.sdp)

    @peer_connection.on("track")
    def on_track(track):
        print("Received new track")
        if track.kind == "video":
            try:
                asyncio.ensure_future(display_video(config, track))
            except Exception as e:
                print('Error while displaying video:', e)
        else:
            # Discard audio if received
            MediaBlackhole().addTrack(track)

    remote_offer = await signaling_server.retrieve_offer()
    print('received offer')
    if not remote_offer:
        return

    offer = RTCSessionDescription(sdp=remote_offer['sdp'], type=remote_offer['type'])
    await peer_connection.setRemoteDescription(offer)
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)
    await signaling_server.send_answer(peer_connection.localDescription.sdp)
    print('sent answer')

    # Keep loop alive
    while True:
        await asyncio.sleep(1)


def depth_to_points(
    depth: np.ndarray,  # [H, W]
    depth_scale: float, # scalar
    K: np.ndarray,      # [3, 3]
    viewmat: np.ndarray # [4, 4]
):
    H, W = depth.shape
    # y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
    # y = y.flatten()
    # x = x.flatten()
    # z = depth.flatten()
    y, x = np.mgrid[:H, :W]
    y = y.flatten()
    x = x.flatten()
    z = (depth.flatten() * depth_scale).astype(np.float32)
    x = (x - K[0, 2]) * z / K[0, 0]
    y = (y - K[1, 2]) * z / K[1, 1]
    ones = np.ones_like(z)
    points = np.stack([x, y, z, ones], axis=0) # [4, H*W]
    points = viewmat @ points # [4, H*W]
    points = points[:3] / points[3]
    points = points.T.reshape(H, W, 3)

    return points


async def display_video(config, track: VideoStreamTrack):
    print("Starting video display...")
    orig_Ks = torch.from_numpy(np.array([[
        [1596.9842529296875, 0, 717.5390625],
        [0, 1596.9842529296875, 956.87127685546875],
        [0, 0, 1]
    ]])).cuda().float()
    orig_W, orig_H = 1440, 1920
    first_frame = True
    render_fn = hydra.utils.instantiate(config.render_fn)
    time_idx = 0
    config['tracking_factor'] = 1
    config['densify_factor'] = 1
    config['mapping_factor'] = 1

    # while True:
    for time_idx in range(100_000):
        curr = time.time()
        try:
            frame = await track.recv()
        except Exception as e:
            print('Error while receiving frame:', e)
            break
        img_bgr = frame.to_ndarray(format="bgr24")  # Convert frame to BGR image
        H, W, C = img_bgr.shape
        W = W // 2
        depth_bgr, color_bgr = img_bgr[:, :W], img_bgr[:, W:]
        color = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        depth = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2HSV)[:, :, [0]]
        depth = depth / 179 * 3.0
        down_factor = orig_W / W
        print(color.shape, depth.shape)

        color = torch.from_numpy(color).cuda().float().permute(2, 0, 1) / 255.0
        depth = torch.from_numpy(depth).cuda().float().permute(2, 0, 1)
        intrinsics = orig_Ks[0].cuda().float() / down_factor

        if first_frame:
            params, variables, first_frame_w2c, cam = initialize_first_timestep(
                # dataset,
                color, # (C, H, W)
                depth, # (C, H, W)
                intrinsics,
                torch.eye(4, dtype=torch.float32, device='cuda'),
                2000,
                config['scene_radius_depth_ratio'],
                config['densify_factor'],
                config['mean_sq_dist_method'],
                config['gaussian_distribution'],
            )

            # Initialize list to keep track of Keyframes
            keyframe_list = []

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
            first_frame = False

        cam = Camera(
            orig_Ks * H / orig_H,
            W,
            H,
        )
        viewer.lock.acquire()
        tic = time.time()
        tracking_metrics, mapping_metrics = slam_frame(
            config,
            render_fn,
            params,
            variables,
            time_idx,
            color,
            depth,
            cam,
            keyframe_list,
            first_frame_w2c,
        )
        # cv2.imshow("WebRTC Stream", img_bgr)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
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


@hydra.main(version_base=None, config_path='../configs', config_name='demo')
def main(config: DictConfig):
    remote_address = 'http://10.42.0.87'

    print("Connecting to:", remote_address)
    asyncio.run(start_receiving_stream(config, remote_address))


if __name__ == "__main__":
    main()