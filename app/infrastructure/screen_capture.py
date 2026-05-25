import os
import time
import subprocess
import threading
import logging
import tempfile
from PIL import ImageGrab

logger = logging.getLogger(__name__)


class ScreenCaptureDaemon:
    def __init__(self, output_path: str = None, interval: float = 5.0):
        """
        Background observer daemon that captures screenshots.
        Designed to support Wayland (via grim), X11 (via Pillow), and mock fallbacks.
        """
        if output_path is None:
            output_path = os.path.join(tempfile.gettempdir(), "screen.png")
        self.output_path = output_path
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self, callback):
        """
        Starts the screen capture loop inside a daemon thread.
        """
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(callback,), daemon=True)
        self.thread.start()
        logger.info("Screen capture background daemon started.")

    def stop(self):
        """
        Stops the screen capture loop.
        """
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        logger.info("Screen capture background daemon stopped.")

    def _run(self, callback):
        while self.running:
            try:
                # 1. Try capturing with grim (Wayland default)
                if self._capture_grim():
                    callback(self.output_path)
                # 2. Try capturing with Pillow ImageGrab (fallback)
                elif self._capture_pillow():
                    callback(self.output_path)
                # 3. Fail-safe mock callback if running in headless environments
                else:
                    callback(None)
            except Exception as e:
                logger.error(f"Error in capture daemon execution: {e}")
            time.sleep(self.interval)

    def _capture_grim(self) -> bool:
        try:
            res = subprocess.run(["grim", self.output_path], capture_output=True, timeout=3)
            return res.returncode == 0
        except Exception:
            return False

    def _capture_pillow(self) -> bool:
        try:
            img = ImageGrab.grab()
            img.save(self.output_path)
            return True
        except Exception:
            return False
