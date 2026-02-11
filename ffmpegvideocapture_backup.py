import subprocess
import numpy as np
import time
import logging
import threading


class FFmpegVideoCapture:
    """
    FFmpeg-based video capture for RTSP streams on Jetson.
    Provides near-live frames with minimal buffering.
    Automatically restarts FFmpeg if frames are missing or the process dies.
    """
    def __init__(self, rtsp_url: str, width=1280, height=960, timeout=5, max_buffer=3, fps=None):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.channels = 3
        self.frame_size = self.width * self.height * self.channels
        self.timeout = timeout
        self.max_buffer = max_buffer
        self.fps = fps
        self.restart_count = 0
        self.proc = None
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stderr_thread = None
        self._start_ffmpeg()

    def _consume_stderr(self):
        """Consume stderr in a separate thread to prevent blocking."""
        while not self._stop_event.is_set():
            if self.proc and self.proc.stderr:
                try:
                    line = self.proc.stderr.readline()
                    if line:
                        logging.debug(f"FFmpeg: {line.decode('utf-8', errors='ignore').strip()}")
                except Exception as e:
                    logging.debug(f"Error reading stderr: {e}")
                    break
            else:
                time.sleep(0.1)

    def _start_ffmpeg(self):
        logging.warning("Starting FFmpeg process...")
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-stimeout", str(int(self.timeout*1e6)),  # microseconds
            "-i", self.rtsp_url,
            "-fflags", "nobuffer",  # Reduce buffering
            "-flags", "low_delay",   # Low latency mode
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-an",  # No audio
        ]
        
        if self.fps:
            cmd.extend(["-r", str(self.fps)])
        
        cmd.append("-")
        
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_size * self.max_buffer
        )
        
        # Start stderr consumer thread
        if self._stderr_thread is None or not self._stderr_thread.is_alive():
            self._stderr_thread = threading.Thread(target=self._consume_stderr, daemon=True)
            self._stderr_thread.start()
        
        self.last_frame_time = time.time()
        
        # Give FFmpeg a moment to initialize
        time.sleep(0.5)

    def _restart_ffmpeg(self):
        logging.error("Restarting FFmpeg...")
        self.restart_count += 1
        with self.lock:
            try:
                if self.proc:
                    self.proc.terminate()
                    self.proc.wait(timeout=2)
            except Exception:
                try:
                    if self.proc:
                        self.proc.kill()
                except Exception:
                    pass
            self.proc = None
            time.sleep(1)  # Reduced from 2 seconds
            self._start_ffmpeg()

    def read(self):
        """Read a single frame. Returns (ret, frame)."""
        if self._stop_event.is_set():
            return False, None

        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                logging.error("FFmpeg process died.")
                self._restart_ffmpeg()
                return False, None

            try:
                # Read the exact frame size
                raw = self.proc.stdout.read(self.frame_size)
                
                if len(raw) != self.frame_size:
                    logging.error(f"FFmpeg incomplete frame: got {len(raw)} bytes, expected {self.frame_size}.")
                    self._restart_ffmpeg()
                    return False, None
                
                self.last_frame_time = time.time()
                frame = np.frombuffer(raw, np.uint8).reshape(
                    (self.height, self.width, self.channels)
                )
                return True, frame
                
            except Exception as e:
                logging.error(f"Error reading frame: {e}")
                self._restart_ffmpeg()
                return False, None

    def release(self):
        """Stop FFmpeg and clean up."""
        self._stop_event.set()
        with self.lock:
            if self.proc:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=2)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                self.proc = None