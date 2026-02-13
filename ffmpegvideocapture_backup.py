import subprocess
import numpy as np
import time
import logging
import threading


class FFmpegVideoCapture:
    """
    FFmpeg-based video capture for RTSP streams on Jetson.
    Always provides the LATEST frame, discarding old buffered frames.
    """
    def __init__(self, rtsp_url: str, width=1280, height=960, timeout=10, fps=None):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.channels = 3
        self.frame_size = self.width * self.height * self.channels
        self.timeout = timeout
        self.fps = fps
        self.restart_count = 0
        self.proc = None
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._stderr_thread = None
        self._frame_thread = None
        self.latest_frame = None
        self.frame_ready = threading.Event()
        self.last_successful_read = time.time()
        self._start_ffmpeg()

    def _consume_stderr(self):
        """Consume stderr in a separate thread to prevent blocking."""
        while not self._stop_event.is_set():
            if self.proc and self.proc.stderr:
                try:
                    line = self.proc.stderr.readline()
                    if line:
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        # Log important messages
                        if any(x in line_str.lower() for x in ['error', 'warning', 'connection', 'timeout']):
                            logging.warning(f"FFmpeg: {line_str}")
                        else:
                            logging.debug(f"FFmpeg: {line_str}")
                except Exception:
                    break
            else:
                time.sleep(0.1)

    def _read_frames_continuously(self):
        """Continuously read frames in background, keeping only the latest."""
        consecutive_failures = 0
        max_failures = 10
        
        while not self._stop_event.is_set():
            if self.proc is None or self.proc.poll() is not None:
                logging.warning("FFmpeg process not running in read thread")
                time.sleep(1)
                continue
                
            try:
                # Use a more robust read with smaller chunks
                raw = b""
                read_timeout = time.time() + self.timeout
                
                while len(raw) < self.frame_size and time.time() < read_timeout:
                    if self._stop_event.is_set():
                        return
                    
                    chunk_size = min(65536, self.frame_size - len(raw))  # Read in 64KB chunks
                    try:
                        chunk = self.proc.stdout.read(chunk_size)
                        if not chunk:
                            logging.warning("FFmpeg stdout returned empty chunk (EOF or disconnected)")
                            break
                        raw += chunk
                    except Exception as e:
                        logging.error(f"Error reading chunk: {e}")
                        break
                
                if len(raw) != self.frame_size:
                    consecutive_failures += 1
                    if len(raw) > 0:
                        logging.warning(f"Incomplete frame: {len(raw)}/{self.frame_size} bytes (failure {consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logging.error("Too many consecutive failures, restarting FFmpeg...")
                        self._restart_ffmpeg()
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                # Successfully read a frame
                consecutive_failures = 0
                frame = np.frombuffer(raw, np.uint8).reshape(
                    (self.height, self.width, self.channels)
                )
                
                # Store only the latest frame (discard old ones)
                with self.lock:
                    self.latest_frame = frame
                    self.frame_ready.set()
                    self.last_successful_read = time.time()
                    
            except Exception as e:
                consecutive_failures += 1
                logging.error(f"Error in frame reading thread: {e}")
                time.sleep(0.1)

    def _start_ffmpeg(self):
        logging.warning("Starting FFmpeg process...")
        
        # Kill any existing process first
        with self.lock:
            if self.proc:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=1)
                except Exception:
                    pass
        
        # Simplified, compatible command
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-f", "rawvideo",
            "-vf", "scale=in_range=pc:out_range=pc",
            "-pix_fmt", "bgr24",
            "-an",  # No audio
        ]
        
        if self.fps:
            cmd.extend(["-r", str(self.fps)])
        
        cmd.append("-")
        
        logging.info(f"FFmpeg command: {' '.join(cmd)}")
        
        with self.lock:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_size * 2  # Small buffer
            )
            self.latest_frame = None
            self.frame_ready.clear()
        
        # Start stderr consumer thread
        if self._stderr_thread is None or not self._stderr_thread.is_alive():
            self._stderr_thread = threading.Thread(target=self._consume_stderr, daemon=True)
            self._stderr_thread.start()
        
        # Start frame reading thread
        if self._frame_thread is None or not self._frame_thread.is_alive():
            self._frame_thread = threading.Thread(target=self._read_frames_continuously, daemon=True)
            self._frame_thread.start()
        
        # Wait for first successful frame with feedback
        logging.info("Waiting for first frame from FFmpeg...")
        for i in range(20):  # Wait up to 10 seconds
            if self.frame_ready.is_set():
                logging.info("First frame received successfully!")
                return
            time.sleep(0.5)
        
        logging.warning("FFmpeg started but first frame not received yet (may still be connecting)")

    def _restart_ffmpeg(self):
        logging.error(f"Restarting FFmpeg... (restart #{self.restart_count + 1})")
        self.restart_count += 1
        
        old_proc = None
        with self.lock:
            old_proc = self.proc
            self.proc = None
            
        try:
            if old_proc:
                old_proc.terminate()
                old_proc.wait(timeout=2)
        except Exception:
            try:
                if old_proc:
                    old_proc.kill()
            except Exception:
                pass
        
        time.sleep(2)
        self._start_ffmpeg()

    def read(self):
        """
        Read the LATEST available frame.
        This always returns the most recent frame, skipping any buffered old frames.
        Returns (ret, frame).
        """
        if self._stop_event.is_set():
            return False, None

        # Check if we've received any frames recently
        if time.time() - self.last_successful_read > 30:
            logging.error("No frames received in 30 seconds, restarting...")
            self._restart_ffmpeg()
            return False, None

        # Wait for a frame to be available
        if not self.frame_ready.wait(timeout=self.timeout):
            logging.error("Timeout waiting for frame")
            return False, None

        with self.lock:
            if self.latest_frame is None:
                return False, None
            # Return a copy of the latest frame
            frame = self.latest_frame.copy()
        
        return True, frame

    def release(self):
        """Stop FFmpeg and clean up."""
        logging.info("Releasing FFmpeg capture...")
        self._stop_event.set()
        
        # Wait for threads to finish
        if self._frame_thread and self._frame_thread.is_alive():
            self._frame_thread.join(timeout=2)
        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=2)
        
        with self.lock:
            proc = self.proc
            self.proc = None
            
        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass