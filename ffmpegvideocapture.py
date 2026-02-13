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
        
        logging.info(f"Initializing FFmpeg capture: {width}x{height}, frame_size={self.frame_size} bytes")
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

    def _read_exact_bytes(self, n_bytes, timeout_seconds):
        """Read exact number of bytes with timeout, or return None if failed."""
        raw = b""
        deadline = time.time() + timeout_seconds
        
        while len(raw) < n_bytes:
            if self._stop_event.is_set():
                return None
            
            if time.time() > deadline:
                logging.error(f"Timeout reading bytes: got {len(raw)}/{n_bytes}")
                return None
            
            if self.proc is None or self.proc.poll() is not None:
                logging.error("FFmpeg process died during read")
                return None
            
            remaining = n_bytes - len(raw)
            chunk_size = min(65536, remaining)
            
            try:
                chunk = self.proc.stdout.read(chunk_size)
                if not chunk:
                    logging.error(f"EOF: got {len(raw)}/{n_bytes} bytes")
                    return None
                raw += chunk
            except Exception as e:
                logging.error(f"Error reading bytes: {e}")
                return None
        
        return raw

    def _discard_bytes(self, n_bytes):
        """Discard n bytes from the stream to resync."""
        logging.warning(f"Discarding {n_bytes} bytes to resync stream...")
        discarded = 0
        while discarded < n_bytes and not self._stop_event.is_set():
            chunk_size = min(65536, n_bytes - discarded)
            try:
                chunk = self.proc.stdout.read(chunk_size)
                if not chunk:
                    logging.error("EOF while discarding bytes")
                    return False
                discarded += len(chunk)
            except Exception as e:
                logging.error(f"Error discarding bytes: {e}")
                return False
        logging.info(f"Discarded {discarded} bytes successfully")
        return True

    def _read_frames_continuously(self):
        """Continuously read frames in background, keeping only the latest."""
        consecutive_failures = 0
        max_failures = 3  # Restart sooner on failures
        frame_count = 0
        
        # Wait a bit for FFmpeg to start outputting
        time.sleep(1)
        
        while not self._stop_event.is_set():
            if self.proc is None or self.proc.poll() is not None:
                logging.warning("FFmpeg process not running in read thread")
                time.sleep(1)
                continue
                
            try:
                # Read exact frame size with timeout
                raw = self._read_exact_bytes(self.frame_size, self.timeout)
                
                if raw is None:
                    consecutive_failures += 1
                    logging.warning(f"Failed to read frame (failure {consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logging.error("Too many consecutive failures, restarting FFmpeg...")
                        self._restart_ffmpeg()
                        consecutive_failures = 0
                        frame_count = 0
                    
                    time.sleep(0.5)
                    continue
                
                if len(raw) != self.frame_size:
                    # CRITICAL: We got partial data. This will corrupt alignment!
                    # We MUST discard this data and skip to the next frame boundary
                    logging.error(f"Got {len(raw)}/{self.frame_size} bytes - DISCARDING to prevent misalignment")
                    
                    # Discard the remainder to get back to frame boundary
                    bytes_to_discard = self.frame_size - len(raw)
                    if not self._discard_bytes(bytes_to_discard):
                        logging.error("Failed to resync, restarting FFmpeg")
                        self._restart_ffmpeg()
                        consecutive_failures = 0
                        frame_count = 0
                    
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        self._restart_ffmpeg()
                        consecutive_failures = 0
                        frame_count = 0
                    
                    continue
                
                # Successfully read a complete frame
                consecutive_failures = 0
                frame_count += 1
                
                try:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width, self.channels)
                    )
                    
                    # Diagnostic: check first few frames colors
                    if frame_count <= 3:
                        sample = frame[self.height//2:self.height//2+10, self.width//2:self.width//2+10, :]
                        logging.info(f"Frame {frame_count} sample (BGR): B={sample[:,:,0].mean():.1f}, G={sample[:,:,1].mean():.1f}, R={sample[:,:,2].mean():.1f}")
                    
                except ValueError as e:
                    logging.error(f"Error reshaping frame: {e}, bytes={len(raw)}, expected={self.frame_size}")
                    consecutive_failures += 1
                    continue
                
                # Store only the latest frame (discard old ones)
                with self.lock:
                    self.latest_frame = frame
                    self.frame_ready.set()
                    self.last_successful_read = time.time()
                    
            except Exception as e:
                consecutive_failures += 1
                logging.error(f"Error in frame reading thread: {e}", exc_info=True)
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
        
        # Build command - ensure correct pixel format and size
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", self.rtsp_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",  # OpenCV expects BGR
            "-s", f"{self.width}x{self.height}",  # Force exact output size
        ]
        
        # Add fps filter if specified
        if self.fps:
            cmd.extend(["-r", str(self.fps)])
        
        cmd.extend([
            "-an",  # No audio
            "-"
        ])
        
        logging.info(f"FFmpeg command: {' '.join(cmd)}")
        
        with self.lock:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0  # Unbuffered - critical for alignment
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
        for i in range(30):  # Wait up to 15 seconds
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
            self.frame_ready.clear()
            self.latest_frame = None
            
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
            # Return the latest frame
            frame = self.latest_frame
        
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