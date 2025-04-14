import asyncio
import time
import aiohttp
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole
import open3d as o3d


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

async def start_receiving_stream(server_url):
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
                asyncio.ensure_future(display_video(track))
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


async def display_video(track: VideoStreamTrack):
    print("Starting video display...")
    orig_K = np.array([
        [1596.9842529296875, 0, 717.5390625],
        [0, 1596.9842529296875, 956.87127685546875],
        [0, 0, 1]
    ])
    orig_W, orig_H = 1440, 1920

    while True:
        curr = time.time()
        frame = await track.recv()
        img_bgr = frame.to_ndarray(format="bgr24")  # Convert frame to BGR image
        H, W, C = img_bgr.shape
        W = W // 2
        depth_bgr, color_bgr = img_bgr[:, :W], img_bgr[:, W:]
        depth = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2HSV)[:, :, 0]
        mask = (0 < depth) & (depth < 179)
        depth = depth / 179 * 3.0
        down_factor = orig_W // W
        print('depth', depth.min(), depth.max())

        # display pointcloud in open3d
        points = depth_to_points(depth, 1, orig_K / down_factor, np.eye(4))
        print(points.shape, color_bgr.shape, mask.shape)
        points = points[mask]
        colors = color_bgr[mask]
        colors = colors[:, ::-1] / 255
        print(points.shape, colors.shape, mask.shape)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])

        print(f'{time.time() - curr:.2f} {img_bgr.shape} {down_factor}')
        cv2.imshow("WebRTC Stream", img_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    remote_address = 'http://10.42.0.87'

    print("Connecting to:", remote_address)
    asyncio.run(start_receiving_stream(remote_address))
