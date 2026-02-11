import time
from collections import deque

class FPSMonitor:
    def __init__(self, window_sec=5.0):
        self.window_sec = window_sec
        
        self.recv_timestamps = deque()
        self.proc_timestamps = deque()
        
        self.latencies = deque()

    def _cleanup(self, dq):
        now = time.time()
        while dq and (now - dq[0] > self.window_sec):
            dq.popleft()

    # ---- FRAME RECEIVED ----
    def frame_received(self):
        now = time.time()
        self.recv_timestamps.append(now)
        self._cleanup(self.recv_timestamps)

    # ---- FRAME PROCESSED ----
    def frame_processed(self, recv_time):
        now = time.time()
        self.proc_timestamps.append(now)
        self._cleanup(self.proc_timestamps)

        latency = now - recv_time
        self.latencies.append(latency)
        if len(self.latencies) > 200:
            self.latencies.popleft()

    def get_stats(self):
        self._cleanup(self.recv_timestamps)
        self._cleanup(self.proc_timestamps)

        recv_fps = len(self.recv_timestamps) / self.window_sec
        proc_fps = len(self.proc_timestamps) / self.window_sec

        avg_latency = (
            sum(self.latencies) / len(self.latencies)
            if self.latencies else 0.0
        )

        return {
            "recv_fps": round(recv_fps, 2),
            "proc_fps": round(proc_fps, 2),
            "avg_latency_ms": round(avg_latency * 1000, 2)
        }
