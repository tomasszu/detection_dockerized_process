import subprocess
import numpy as np

class FFmpegVideoCapture:
    def __init__(self, rtsp_url: str):
        # Run ffmpeg as a subprocess, decode to raw BGR frames
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-rtsp_transport", "tcp",   # force TCP (important for RTSP stability)
                "-i", rtsp_url,
                "-f", "rawvideo",           # output raw video frames
                "-pix_fmt", "bgr24",        # in OpenCV format
                "-vcodec", "rawvideo",
                "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

        # TODO: set these from stream metadata instead of hardcoding
        self.width = 1280
        self.height = 960
        self.channels = 3
        self.frame_size = self.width * self.height * self.channels

    def read(self):
        raw = self.proc.stdout.read(self.frame_size)
        if len(raw) != self.frame_size:
            return False, None
        frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, self.channels))
        return True, frame

    def release(self):
        if self.proc:
            self.proc.kill()
            self.proc = None
