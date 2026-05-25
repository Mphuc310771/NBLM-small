import os
import logging
import asyncio
import threading
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.presentation.api import router as api_router, vector_store
from app.application.memory_daemon import MemoryDaemon
from app.infrastructure.vision_adapter import VisionAdapter

# Configure logging for the entire app
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)

# Initialize Memory Daemon (Producer) and Vision Adapter (Consumer)
memory_daemon = MemoryDaemon()
vision_adapter = VisionAdapter()


def sys_monitor_daemon():
    """
    Lightweight background thread that periodically logs system status (X11/Wayland/Linux state)
    to check system integrity.
    """
    while True:
        try:
            session_type = os.environ.get("XDG_SESSION_TYPE", "unknown")
            display = os.environ.get("DISPLAY", "none")
            wayland_display = os.environ.get("WAYLAND_DISPLAY", "none")
            logging.info(
                f"[SYSTEM STATE MONITOR] status=HEALTHY session={session_type} "
                f"display={display} wayland={wayland_display}"
            )
        except Exception as e:
            logging.error(f"[SYSTEM STATE MONITOR] Error: {e}")
        time.sleep(30)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start screen memory capture daemon
    memory_daemon.start()
    
    # Start async event consumer task for gRPC processing
    asyncio.create_task(vision_adapter.start_event_consumer(vector_store))
    
    # Start system monitor thread
    monitor_thread = threading.Thread(target=sys_monitor_daemon, daemon=True)
    monitor_thread.start()
    logging.info("Startup complete: Background daemons and async event consumer active.")
    
    yield
    
    # Gracefully stop capture daemon
    memory_daemon.stop()
    logging.info("Shutdown complete: Background daemons stopped.")


app = FastAPI(
    title="RAG API Hub",
    description="A FastAPI RAG application built using Clean Architecture principles.",
    version="1.0.0",
    lifespan=lifespan
)


# Include the API presentation router (MUST be registered before StaticFiles mount)
app.include_router(api_router)

# Serve the frontend UI from the static directory.
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/")
def root():
    """Redirect root to the frontend UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")
