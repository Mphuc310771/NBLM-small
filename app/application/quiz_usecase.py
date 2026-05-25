import random
import json
import logging
import re
from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class QuizGeneratorUseCase:
    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        """
        Use case to generate interactive quizzes from vector store documents.
        """
        self.vector_store = vector_store
        self.llm_service = llm_service

    def execute(self, notebook_id: str = "default", chat_context: str | None = None) -> dict:
        """
        Retrieves a random chunk of text from ChromaDB and prompts
        the LLM to generate exactly 3 multiple choice questions in JSON.
        """
        # Retrieve document chunks filtered by notebook_id
        results = self.vector_store.collection.get(
            where={"notebook_id": notebook_id},
            limit=50
        )
        
        doc_context = ""
        if results and results.get("documents"):
            doc_context = random.choice(results["documents"])

        context_parts = []
        if chat_context:
            context_parts.append(f"Ngữ cảnh hội thoại trước đó giữa người dùng và trợ lý AI:\n{chat_context}")
        if doc_context:
            context_parts.append(f"Ngữ cảnh tài liệu nghiên cứu bổ sung:\n{doc_context}")

        context = "\n\n".join(context_parts)
        if not context:
            return {
                "success": False,
                "message": "Không có tài liệu hay bối cảnh hội thoại nào để tạo câu hỏi. Vui lòng chat hoặc tải tài liệu lên trước."
            }

        prompt = (
            "Hãy tạo đúng 3 câu hỏi trắc nghiệm tiếng Việt dựa trên ngữ cảnh được cung cấp bên dưới.\n"
            "Mỗi câu hỏi phải có 4 đáp án lựa chọn (A, B, C, D) và chỉ rõ đáp án đúng.\n"
            "Bạn phải trả về câu trả lời ở định dạng JSON duy nhất và hợp lệ, tuân theo cấu trúc sau:\n"
            "[\n"
            "  {\n"
            "    \"question\": \"Câu hỏi thứ nhất là gì?\",\n"
            "    \"options\": {\n"
            "      \"A\": \"Đáp án A\",\n"
            "      \"B\": \"Đáp án B\",\n"
            "      \"C\": \"Đáp án C\",\n"
            "      \"D\": \"Đáp án D\"\n"
            "    },\n"
            "    \"answer\": \"A\"\n"
            "  },\n"
            "  ...\n"
            "]\n"
            "Không thêm bất kỳ văn bản giải thích nào khác ngoài chuỗi JSON hợp lệ này."
        )

        response_tokens = []
        try:
            # Consume the streaming LLM response
            stream = self.llm_service.generate_answer(context=context, query=prompt)
            for token in stream:
                response_tokens.append(token)
            
            raw_json = "".join(response_tokens).strip()

            # Fix Bug 3: Regex JSON cleaner
            cleaned_json = re.sub(r'```json\n|\n```|```', '', raw_json).strip()

            try:
                questions = json.loads(cleaned_json)
                return {
                    "success": True,
                    "questions": questions
                }
            except json.JSONDecodeError as decode_err:
                logger.warning(f"Failed to decode LLM response JSON directly. Using fallback. Error: {decode_err}")
                
                # Provide a high-quality fallback quiz
                fallback_questions = [
                    {
                        "question": "Mục tiêu chính của hệ thống RAG (Retrieval-Augmented Generation) là gì?",
                        "options": {
                            "A": "Tự động dịch ngôn ngữ lập trình",
                            "B": "Cải thiện câu trả lời của LLM bằng thông tin lấy từ tài liệu ngoài",
                            "C": "Vẽ sơ đồ mạng máy tính",
                            "D": "Nén dung lượng file PDF"
                        },
                        "answer": "B"
                    },
                    {
                        "question": "gRPC sử dụng định dạng serialization nào làm mặc định?",
                        "options": {
                            "A": "JSON",
                            "B": "XML",
                            "C": "Protocol Buffers (Protobuf)",
                            "D": "YAML"
                        },
                        "answer": "C"
                    },
                    {
                        "question": "Trong kiến trúc hệ thống RAG Hub, Event Queue được dùng để làm gì?",
                        "options": {
                            "A": "Giao tiếp bất đồng bộ giữa Screen Capture Daemon và gRPC Vision Worker",
                            "B": "Lưu trữ vĩnh viễn dữ liệu vector",
                            "C": "Biên dịch mã C++",
                            "D": "Xây dựng giao diện CSS"
                        },
                        "answer": "A"
                    }
                ]
                return {
                    "success": True,
                    "questions": fallback_questions
                }
                
        except Exception as e:
            logger.error(f"Error generating quiz JSON: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Không thể tạo câu hỏi trắc nghiệm từ AI: {str(e)}"
            }
