import os
import time
import logging
import asyncio
from app.infrastructure.screen_capture import ScreenCaptureDaemon
from app.core.events import event_queue

logger = logging.getLogger(__name__)


class MemoryDaemon:
    def __init__(self, loop: asyncio.AbstractEventLoop = None):
        """
        Memory Daemon acting purely as a PRODUCER in the distributed architecture.
        It monitors the user screen and publishes a SCREEN_CAPTURED event to the queue.
        """
        self.capture_daemon = ScreenCaptureDaemon()
        self.loop = loop

    def start(self):
        """
        Starts the background capture thread.
        """
        if not self.loop:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                self.loop = asyncio.get_event_loop()
        
        self.capture_daemon.start(callback=self._handle_screenshot)

    def stop(self):
        """
        Stops the capture thread.
        """
        self.capture_daemon.stop()

    def _handle_screenshot(self, image_path: str):
        """
        Pushes a SCREEN_CAPTURED event containing raw image bytes to the shared event queue.
        """
        try:
            image_data = b""
            filename = "screenshot.png"
            
            # Read image file bytes if valid, fallback to mock bytes if headless
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    image_data = f.read()
                filename = os.path.basename(image_path)
            else:
                image_data = b"MOCK_SCREENSHOT_DATA"

            event = {
                "type": "SCREEN_CAPTURED",
                "image_data": image_data,
                "filename": filename,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # Enqueue the event thread-safely from background thread
            self.loop.call_soon_threadsafe(event_queue.put_nowait, event)
            logger.info("MemoryDaemon (Producer): Published SCREEN_CAPTURED event to queue.")
            
        except Exception as e:
            logger.error(f"MemoryDaemon Producer failed to publish event: {e}")
