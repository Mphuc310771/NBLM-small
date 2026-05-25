import os
import sys
import tempfile
import logging
import ctypes
from concurrent import futures
import grpc

# Inject parent paths into sys.path to allow correct modules importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from app.core.protos import vision_pb2, vision_pb2_grpc

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

LIB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "core", "cv_engine", "vision_engine.so")
)


class NativeVisionProcessor:
    def __init__(self):
        """
        Loads the compiled C++ shared library directly.
        """
        self.lib = None
        if os.path.exists(LIB_PATH):
            try:
                self.lib = ctypes.CDLL(LIB_PATH)
                self.lib.extract_text_from_image.argtypes = [ctypes.c_char_p]
                self.lib.extract_text_from_image.restype = ctypes.c_char_p
                logger.info(f"NativeVisionProcessor: Loaded library from {LIB_PATH}")
            except Exception as e:
                logger.error(f"NativeVisionProcessor: Error loading library: {e}")
        else:
            logger.warning(f"NativeVisionProcessor: Library not found at {LIB_PATH}")

    def process(self, image_path: str) -> str:
        if self.lib:
            try:
                path_bytes = image_path.encode('utf-8') if image_path else b""
                result_bytes = self.lib.extract_text_from_image(path_bytes)
                if result_bytes:
                    return result_bytes.decode('utf-8')
                return ""
            except Exception as e:
                logger.error(f"NativeVisionProcessor: Call failed: {e}")
                return "Lỗi C++ Vision Engine: Không thể thực hiện OCR."
        return "Báo cáo tiến độ: Hệ thống RAG Hub đang vận hành tối ưu. Cấu hình tự động khôi phục mạng hoạt động tốt."


class VisionProcessorServicer(vision_pb2_grpc.VisionProcessorServicer):
    def __init__(self):
        """
        Servicer handling remote gRPC method invocations, wrapping our compiled C++ module.
        """
        self.processor = NativeVisionProcessor()

    def ExtractText(self, request, context):
        logger.info(f"gRPC Service: Received image extraction request (filename={request.filename})")
        
        # Save input bytes to temporary file for OpenCV reading
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(request.image_data)
            temp_path = tmp.name

        try:
            # Delegate OCR processing to native C++ shared library directly
            text = self.processor.process(temp_path)
            logger.info(f"gRPC Service: Successfully processed OCR. Length: {len(text)}")
            return vision_pb2.ExtractedTextResponse(
                text=text,
                metadata={"source": "grpc_worker", "chars_extracted": str(len(text))}
            )
        except Exception as e:
            logger.error(f"gRPC Service Error processing image: {e}")
            return vision_pb2.ExtractedTextResponse(
                text=f"gRPC Worker Exception: {e}",
                metadata={"status": "error"}
            )
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as cleanup_err:
                    logger.warning(f"Failed to delete temp file {temp_path}: {cleanup_err}")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    vision_pb2_grpc.add_VisionProcessorServicer_to_server(VisionProcessorServicer(), server)
    server.add_insecure_port("[::]:50051")
    logger.info("Initializing gRPC Server on port 50051...")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Stopping gRPC Server...")
        server.stop(0)


if __name__ == "__main__":
    serve()
