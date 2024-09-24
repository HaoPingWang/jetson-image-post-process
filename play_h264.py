# -*- coding: utf-8 -*-
#import pycuda.driver as cuda
import vpi
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

use_vpi = True
use_udpsrc = True

VIDEO_LOCATION = "\"/home/oomii/Downloads/gravity_2k-trailer/Gravity - 2K Trailer.mp4\""


gst_src = []
if use_udpsrc:
    gst_src = ["udpsrc port=7001", "application/x-rtp", "rtph264depay"]
    W, H = 1024, 600
else:
    gst_src = [f"filesrc location={VIDEO_LOCATION}", "qtdemux"]
    W, H = 2048, 858
    
get_frame_elements = gst_src + [
    "queue", "h264parse", "nvv4l2decoder", "nvvidconv",
    "videorate", 
    f"video/x-raw,format=RGBA, width={W}, height={H}",
    f"appsink name=appsink0 sync={'false' if use_udpsrc else 'true'} emit-signals=true"
]

display_elements = [
    "appsrc",
    f"video/x-raw,format=RGBA,width={W},height={H},framerate=10/1",
    "nvvidconv",
    "nvoverlaysink display-id=0 sync=false"]


def get_frame(appsink):
    sample = appsink.emit('pull-sample')
    if not sample:
        return
    
    buf = sample.get_buffer()
    caps = sample.get_caps()
    width = caps.get_structure(0).get_value('width')
    height = caps.get_structure(0).get_value('height')

    # 获取缓冲区中的数据
    success, map_info = buf.map(Gst.MapFlags.READ)

    arr = np.ndarray(
        (height, width, 4),
        dtype=np.uint8,
        buffer=map_info.data
    )
    # 解除映射缓冲区
    buf.unmap(map_info)
    return arr

def display_vpi_image(frame: vpi.Image, src):
    # 将 NumPy 数组转换为 GStreamer 缓冲区
    buffer = Gst.Buffer.new_allocate(None, W*H*4, None)
    if type(frame) == vpi.Image:
        with frame.lock():
            # frame.nbytes = 2457600
            frame = frame.cpu()

    buffer.fill(0, frame.tobytes())
    # 设置缓冲区的时间戳
    buffer.pts = Gst.util_uint64_scale(Gst.CLOCK_TIME_NONE, 1, 1)
    buffer.duration = Gst.util_uint64_scale(Gst.CLOCK_TIME_NONE, 1, 1)

    # 推送缓冲区到 src
    ret = src.emit('push-buffer', buffer)
    if ret != Gst.FlowReturn.OK:
        print("Error pushing buffer to src")


def correct_distortion(frame):
    input = vpi.asimage(frame)
    with vpi.Backend.CUDA:
        output = input.remap(warp, interp=vpi.Interp.CATMULL_ROM, border=vpi.Border.ZERO)
    
    return output

grid = vpi.WarpGrid((W, H))
sensorWidth = 22.2 # APS-C sensor
focalLength = 7.5
f = focalLength * W / sensorWidth
 
K = [[f, 0, W/2],
     [0, f, H/2]]
X = np.eye(3,4)
f = focalLength * W / sensorWidth
warp = vpi.WarpMap.fisheye_correction(grid, K=K, X=X,
                                      mapping=vpi.FisheyeMapping.EQUIDISTANT,
                                      coeffs=[-0.1, 0.004])
 
# 初始化GStreamer
Gst.init(None)

get_frame_pipeline = Gst.parse_launch(" ! ".join(get_frame_elements))
display_pipeline = Gst.parse_launch(" ! ".join(display_elements))

get_frame_pipeline.set_state(Gst.State.PLAYING)
display_pipeline.set_state(Gst.State.PLAYING)

appsink = get_frame_pipeline.get_by_name("appsink0")
appsrc = display_pipeline.get_by_name("appsrc0")

frame_count = 0

try:
    while True:
        frame = get_frame(appsink)
        if frame is None:
            print("No frame")
            break

        frame_count += 1
        if use_vpi:
            frame = correct_distortion(frame)  # 进行 VPI 畸变校正

        display_vpi_image(frame, appsrc)

# ctrl + c 退出
except KeyboardInterrupt:
    print("Terminating process")
    get_frame_pipeline.set_state(Gst.State.NULL)
    display_pipeline.set_state(Gst.State.NULL)
