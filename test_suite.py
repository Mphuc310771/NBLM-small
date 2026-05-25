import os
import sys
import unittest
import time
import grpc
from unittest.mock import MagicMock, patch

# Ensure the app directories are importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.core.protos import vision_pb2, vision_pb2_grpc
from app.infrastructure.code_executor import PythonSandbox
from app.application.code_interpreter_usecase import CodeInterpreterUseCase
from app.infrastructure.vision_adapter import VisionAdapter
from app.infrastructure.network_healer import NetworkAutoHealer
from app.application.rag_usecase import RAGUseCase
from app.infrastructure.vector_store import ChromaDBStore
from app.domain.interfaces import ILLMService


class TestDistributedRAGHub(unittest.TestCase):

    def test_code_interpreter_sandbox(self):
        """
        Tests that PythonSandbox successfully executes python code and intercepts matplotlib charts.
        """
        sandbox = PythonSandbox()
        code = (
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "print('Hello Sandbox')\n"
            "plt.plot([1, 2], [3, 4])\n"
            "plt.show()\n"
        )
        res = sandbox.execute(code)
        self.assertTrue(res["success"])
        self.assertIn("Hello Sandbox", res["stdout"])
        self.assertGreater(len(res["charts"]), 0)
        self.assertTrue(res["charts"][0].startswith("/static/outputs/"))

    def test_grpc_vision_pipeline(self):
        """
        Tests the gRPC vision worker server and client communication.
        """
        channel = grpc.insecure_channel("127.0.0.1:50051")
        try:
            # Check connection readiness
            grpc.channel_ready_future(channel).result(timeout=5)
            stub = vision_pb2_grpc.VisionProcessorStub(channel)
            request = vision_pb2.ImageRequest(
                image_data=b"MOCK_BYTES",
                filename="test_img.png"
            )
            response = stub.ExtractText(request, timeout=5)
            self.assertIsNotNone(response.text)
            self.assertIn("source", response.metadata)
            print("[TEST INFO] gRPC OCR Response:", response.text.encode('utf-8', errors='ignore'))
        except (grpc.FutureTimeoutError, grpc.RpcError) as e:
            self.fail(f"gRPC Vision Server on port 50051 is not responding. Error: {e}")

    @patch("app.infrastructure.groq_adapter.Groq")
    def test_rag_network_self_healing(self, mock_groq_class):
        """
        Tests that a Groq rate-limiting or network error triggers the Self-Healing process.
        """
        from groq import APIConnectionError
        
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        
        # Setup mock chunks for the stream iterator
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock(delta=MagicMock(content="Answer after self-healing.", tool_calls=None))]
        
        # GroqAdapter retries internally 3 times before raising the exception.
        # To bubble the exception up to RAGUseCase and trigger the auto-healer,
        # we need 3 failures, followed by a success on the next attempt after healing.
        mock_client.chat.completions.create.side_effect = [
            APIConnectionError(message="Groq connection dropped. Attempt 1", request=MagicMock()),
            APIConnectionError(message="Groq connection dropped. Attempt 2", request=MagicMock()),
            APIConnectionError(message="Groq connection dropped. Attempt 3", request=MagicMock()),
            [mock_chunk]
        ]
        
        # Mock vector store
        mock_store = MagicMock()
        mock_store.search_similar.return_value = [{"text": "Context chunk", "metadata": {"source": "doc"}}]
        
        # Mock healer
        with patch("app.infrastructure.network_healer.NetworkAutoHealer.heal") as mock_heal:
            mock_heal.return_value = True
            
            from app.infrastructure.groq_adapter import GroqAdapter
            llm_service = GroqAdapter(api_key="mock_key")
            rag = RAGUseCase(vector_store=mock_store, llm_service=llm_service)
            
            generator = rag.execute("Test query")
            events = list(generator)
            
            # Assert self-healing alerts were triggered and execution resumed
            alert_triggered = any(e.get("type") == "alert" for e in events)
            success_triggered = any(e.get("type") == "success" for e in events)
            
            self.assertTrue(alert_triggered, "Alert event was not triggered on network failure")
            self.assertTrue(success_triggered, "Success event was not triggered after self-healing completion")

    def test_document_deletion(self):
        """
        Tests that DeleteUseCase calls delete_document on vector store.
        """
        mock_store = MagicMock()
        from app.application.delete_usecase import DeleteUseCase
        use_case = DeleteUseCase(vector_store=mock_store)
        res = use_case.execute("test.pdf")
        
        self.assertTrue(res["success"])
        mock_store.delete_document.assert_called_once_with("test.pdf", "default")

    def test_fallback_llm_service(self):
        """
        Tests that FallbackLLMService falls back through the hierarchy:
        Mistral (fail) -> SambaNova (fail) -> Gemini (fail) -> Groq (success).
        """
        mock_mistral = MagicMock()
        mock_mistral.generate_answer.side_effect = Exception("Mistral Error")
        
        mock_sambanova = MagicMock()
        mock_sambanova.generate_answer.side_effect = Exception("SambaNova Error")
        
        mock_gemini = MagicMock()
        mock_gemini.generate_answer.side_effect = Exception("Gemini Error")
        
        mock_groq = MagicMock()
        mock_groq.generate_answer.return_value = ["Groq Response Chunk"]
        
        from app.infrastructure.fallback_llm import FallbackLLMService
        fallback_service = FallbackLLMService(
            groq_adapter=mock_groq, 
            gemini_adapter=mock_gemini,
            openrouter_adapter=None,
            sambanova_adapter=mock_sambanova,
            mistral_adapter=mock_mistral
        )
        
        tokens = list(fallback_service.generate_answer(context="Context", query="Query"))
        
        self.assertTrue(any("Groq Response Chunk" in t for t in tokens))
        self.assertTrue(any("SambaNova" in t for t in tokens))
        self.assertTrue(any("Gemini" in t for t in tokens))
        mock_mistral.generate_answer.assert_called_once()
        mock_sambanova.generate_answer.assert_called_once()
        mock_gemini.generate_answer.assert_called_once()
        mock_groq.generate_answer.assert_called_once()

    def test_podcast_use_case(self):
        """
        Tests that PodcastUseCase generates dialogue script.
        """
        mock_store = MagicMock()
        mock_store.collection.count.return_value = 5
        mock_store.collection.get.return_value = {
            "documents": ["Chunk 1 content", "Chunk 2 content"],
            "metadatas": [{"source": "test.txt"}, {"source": "test.txt"}]
        }
        
        mock_llm = MagicMock()
        mock_llm.generate_answer.return_value = ['[{"host": "A", "text": "Hello world"}]']
        
        import asyncio
        from unittest.mock import AsyncMock
        
        from app.application.podcast_usecase import PodcastUseCase
        use_case = PodcastUseCase(vector_store=mock_store, llm_service=mock_llm)
        
        with patch("edge_tts.Communicate") as mock_comm:
            mock_comm_inst = MagicMock()
            mock_comm_inst.save = AsyncMock()
            mock_comm.return_value = mock_comm_inst
            
            res = asyncio.run(use_case.execute())
        
        self.assertTrue(res["success"])
        self.assertEqual(len(res["script"]), 1)
        self.assertEqual(res["script"][0]["host"], "A")
        self.assertEqual(res["script"][0]["text"], "Hello world")

    def test_synthesis_use_case(self):
        """
        Tests that SynthesisUseCase compiles notes.
        """
        mock_llm = MagicMock()
        mock_llm.generate_answer.return_value = ["Study Guide Result"]
        
        from app.application.synthesis_usecase import SynthesisUseCase
        use_case = SynthesisUseCase(llm_service=mock_llm)
        res = use_case.execute(notes=["Note 1 text", "Note 2 text"], action="study_guide")
        
        self.assertTrue(res["success"])
        self.assertEqual(res["result"], "Study Guide Result")
        mock_llm.generate_answer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
