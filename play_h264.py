# -*- coding: utf-8 -*-
import vpi
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

VIDEO_LOCATION = "\"/home/oomii/Downloads/gravity_2k-trailer/Gravity - 2K Trailer.mp4\""
DECODE_ELEMENTS = ["h264parse", "queue", "nvv4l2decoder", "nvvidconv", "video/x-raw, format=RGBA"]

class jetson_video_bridge():
    def __init__(self, id, port=None):
        self.frame = None
        if not port:
            self.W, self.H = 2048, 858
            sync = True
            gst_src = [f"filesrc location={VIDEO_LOCATION}", "qtdemux"]
        else:
            self.W, self.H = 1024, 600
            sync = False
            gst_src = [f"udpsrc port={port}", "application/x-rtp", "rtph264depay"]
            
        get_frame_elements = gst_src + DECODE_ELEMENTS
        get_frame_elements += ["queue", f"appsink name=appsink{id} sync={sync} emit-signals=true"]

        display_elements = [
            f"appsrc name=appsrc{id}",
            f"video/x-raw, format=RGBA, height={self.H}, width={self.W}, framerate=30/1",
            f"nvvidconv", f"nvoverlaysink name=nvoverlaysink{id} display-id={id} sync=false"
        ]

        self.get_frame_pipeline = Gst.parse_launch(" ! ".join(get_frame_elements))
        self.display_pipeline = Gst.parse_launch(" ! ".join(display_elements))

        self.appsink = self.get_frame_pipeline.get_by_name(f"appsink{id}")
        self.appsrc = self.display_pipeline.get_by_name(f"appsrc{id}")

        self.set_pipelines_state(Gst.State.PLAYING)
        
        grid = vpi.WarpGrid((self.W, self.H))
        sensorWidth = 22.2 # APS-C sensor
        focalLength = 7.5
        f = focalLength * self.W / sensorWidth
        
        K = [[f, 0, self.W/2],
            [0, f, self.H/2]]
        X = np.eye(3, 4)
        f = focalLength * self.W / sensorWidth
        self.warp = vpi.WarpMap.fisheye_correction(grid, K=K, X=X,
                                            mapping=vpi.FisheyeMapping.EQUIDISTANT,
                                            coeffs=[-0.1, 0.004])
        
    def set_pipelines_state(self, state):
        self.get_frame_pipeline.set_state(state)
        self.display_pipeline.set_state(state)

    def get_frame(self):
        # 清空 appsink 的緩衝區
        while self.appsink.emit('try-pull-sample', 0):
            pass
        
        sample = self.appsink.emit('pull-sample')
        if not sample:
            return
        
        caps = sample.get_caps()
        w = caps.get_structure(0).get_value('width')
        h = caps.get_structure(0).get_value('height')
        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)  # 获取缓冲区中的数据

        self.frame = np.ndarray((h, w, 4), dtype=np.uint8, buffer=map_info.data)
        buf.unmap(map_info)  # 解除映射缓冲区

    def display_vpi_image(self):        
        # 将 NumPy 数组转换为 GStreamer 缓冲区
        buffer = Gst.Buffer.new_allocate(None, self.frame.shape[0]*self.frame.shape[1]*4, None)    
        buffer.fill(0, self.frame.tobytes())
        
        # 设置缓冲区的时间戳
        buffer.pts = Gst.util_uint64_scale(Gst.CLOCK_TIME_NONE, 1, 1)
        buffer.duration = Gst.util_uint64_scale(Gst.CLOCK_TIME_NONE, 1, 1)

        # 推送缓冲区到 src
        ret = self.appsrc.emit('push-buffer', buffer)
        if ret != Gst.FlowReturn.OK:
            print("Error pushing buffer to src")
        
def correct_distortion(frame, warp):
    with vpi.Backend.CUDA:
        vpi_image = vpi.asimage(frame).convert(vpi.Format.NV12_ER)
        vpi_image = vpi_image.remap(warp, interp=vpi.Interp.CATMULL_ROM, border=vpi.Border.ZERO)
        vpi_image = vpi_image.convert(vpi.Format.RGBA8)
    
    with vpi_image.lock():
        frame = vpi_image.cpu()

    return frame


try:
    # 初始化GStreamer
    Gst.init(None)
    left = jetson_video_bridge(0, 7001)
    right = jetson_video_bridge(1)

    left.set_pipelines_state(Gst.State.PLAYING)
    right.set_pipelines_state(Gst.State.PLAYING)

    frame_count = 0

    while True:
        left.get_frame()
        right.get_frame()

        frame_count += 1
        # print(f"Frame count: {frame_count}")
        left.frame = correct_distortion(left.frame, left.warp)
        right.frame = correct_distortion(right.frame, right.warp)
        left.display_vpi_image()
        right.display_vpi_image()

# ctrl + c 退出
finally:
    print("Terminating process")
    left.set_pipelines_state(Gst.State.NULL)
    right.set_pipelines_state(Gst.State.NULL)
