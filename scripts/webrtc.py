import asyncio
import time
import aiohttp
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaBlackhole

class SignalingServer:
    def __init__(self, server_url):
        self.server_url = server_url

    async def retrieve_offer(self):
        url = self.server_url + '/getOffer'
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as resp:
                    return await resp.json()
            except Exception as e:
                print('Error while requesting an offer:', e)
                return None

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
            asyncio.ensure_future(display_video(track))
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
        img = frame.to_ndarray(format="bgr24")  # Convert frame to BGR image
        H, W, C = img.shape
        W = W // 2
        depth_rgb, color_rgb = img[:, :W], img[:, W:]
        depth = 
        down_factor = orig_W // W
        print(f'{time.time() - curr:.2f} {img.shape} {down_factor}')
        cv2.imshow("WebRTC Stream", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    remote_address = 'http://10.42.0.87'

    print("Connecting to:", remote_address)
    asyncio.run(start_receiving_stream(remote_address))
