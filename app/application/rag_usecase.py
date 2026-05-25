import os
import re
import json
import logging
from typing import Generator
from app.domain.interfaces import ILLMService
from app.infrastructure.vector_store import ChromaDBStore

logger = logging.getLogger(__name__)


class RAGUseCase:
    def __init__(self, vector_store: ChromaDBStore, llm_service: ILLMService):
        """
        Orchestrator of the RAG pipeline.
        Manages:
        1. Context Search
        2. Web Fallback Search
        3. Self-Healing Network strategy via Playwright
        4. Tool Calling execution (Code Interpreter & Web Automation)
        """
        self.vector_store = vector_store
        self.llm_service = llm_service

    def execute(self, query: str, provider: str = "auto", notebook_id: str = "default") -> Generator[dict, None, None]:
        """
        Executes the RAG pipeline with built-in network self-healing retries.
        """
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                # Delegate to the main RAG flow
                yield from self._execute_rag(query, provider, notebook_id)
                return  # Successful run, exit loop
            except Exception as e:
                # Capture connection exceptions from Groq or requests
                from groq import APIConnectionError
                import requests
                
                is_network_err = isinstance(e, (APIConnectionError, requests.RequestException)) or "connection" in str(e).lower()

                if is_network_err and attempt < max_attempts:
                    yield {
                        "type": "alert",
                        "content": "🚨 Cảnh báo: Mất kết nối mạng. Đang tự động kích hoạt Playwright để bypass Captive Portal khôi phục Internet..."
                    }
                    
                    try:
                        from app.infrastructure.network_healer import NetworkAutoHealer
                        import asyncio
                        
                        healer = NetworkAutoHealer()
                        
                        # Execute async healing script within sync context
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        healed = loop.run_until_complete(healer.heal())
                        loop.close()

                        if healed:
                            yield {
                                "type": "success",
                                "content": "✅ Đã khôi phục mạng thành công. Tiếp tục xử lý..."
                            }
                            continue  # Retry execution
                        else:
                            yield {
                                "type": "error",
                                "content": "❌ Không thể khôi phục kết nối mạng tự động."
                            }
                            raise e
                    except Exception as heal_err:
                        logger.error(f"Error during self-healing: {heal_err}")
                        yield {
                            "type": "error",
                            "content": f"❌ Lỗi trong quá trình tự sửa lỗi mạng: {heal_err}"
                        }
                        raise e
                else:
                    # Final attempt failed or non-network error
                    yield {
                        "type": "error",
                        "content": f"Lỗi thực thi: {str(e)}"
                    }
                    return

    def _execute_rag(self, query: str, provider: str = "auto", notebook_id: str = "default") -> Generator[dict, None, None]:
        # Step 1: Call vector store to retrieve similar chunks and metadata
        results = self.vector_store.search_similar(query, notebook_id=notebook_id)

        # Assess relevance: distance threshold > 1.3 means very low relevance
        is_low_relevance = False
        if not results:
            is_low_relevance = True
        elif results[0].get("distance", 999.0) > 1.3:
            is_low_relevance = True

        if is_low_relevance:
            # Yield control message indicating internet search fallback
            yield {
                "type": "token",
                "content": "*🌐 Tài liệu không có, đang trích xuất dữ liệu từ Internet...*\n\n"
            }
            # Fetch web search context
            from app.infrastructure.web_search import FallbackWebSearch
            web_search = FallbackWebSearch()
            web_context = web_search.search(query)

            if not web_context:
                web_context = "No results found on the web for this query."
            
            context = f"Web Search Context:\n{web_context}"
            if len(context) > 12000:
                context = context[:12000] + "\n... [TRUNCATED DUE TO TPM LIMITS] ..."
            yield {"type": "citation", "content": [{"source": "Internet Web Search"}]}
            query_prompt = f"The local document didn't have the answer, but I found this on the web:\n{context}\nAnswer the user's query: {query}"
        else:
            citations = [item["metadata"] for item in results if "metadata" in item]
            yield {"type": "citation", "content": citations}
            context = "\n\n".join([item["text"] for item in results])
            if len(context) > 12000:
                context = context[:12000] + "\n... [TRUNCATED DUE TO TPM LIMITS] ..."
            query_prompt = query

        # Math rules and agentic instructions prompt
        system_prompt = (
            "You MUST format all math expressions, Big-O notations, and formulas using strictly LaTeX syntax. "
            "Wrap inline math with single $ (e.g., $O(N^2)$) and block math with double $$.\n"
            "If the user query is about generating a quiz, making a multiple-choice question set, or contains "
            "the phrase 'tạo trắc nghiệm' or 'quiz', you MUST generate exactly 3 multiple choice questions based "
            "on the context. Format the quiz response strictly as a JSON block wrapped in <quiz> and </quiz> tags.\n"
            "If the user asks for mathematical calculations, running data analysis, or drawing charts, you MUST write "
            "Python code and call the execute_python_code tool.\n"
            "If the user asks to download, fetch, or retrieve external secure files/documents from a URL, you MUST "
            "use the web_automation_download tool."
        )

        stream = self.llm_service.generate_answer(context=context, query=query_prompt, system_prompt=system_prompt, provider=provider)
        
        buffer = ""
        tool_call_text = ""
        is_tool_call = False
        
        for token in stream:
            buffer += token
            # Detect tool call triggers
            if "<tool_call" in buffer or "function=execute_python_code" in buffer or "function=web_automation_download" in buffer:
                is_tool_call = True
                
            if is_tool_call:
                # Yield any text that came before the tool call in the buffer
                idx = -1
                if "<tool_call" in buffer:
                    idx = buffer.find("<tool_call")
                elif "function=execute_python_code" in buffer:
                    idx = buffer.find("function=execute_python_code")
                elif "function=web_automation_download" in buffer:
                    idx = buffer.find("function=web_automation_download")
                
                if idx > 0:
                    yield {"type": "token", "content": buffer[:idx]}
                    buffer = buffer[idx:]
                
                tool_call_text = buffer
            else:
                # Check for partial prefix of "<tool_call:" or "function=" to delay yielding
                has_partial = False
                tag = "<tool_call:"
                func_tag = "function="
                
                for i in range(1, len(tag)):
                    if buffer.endswith(tag[:i]) or buffer.endswith(func_tag[:min(i, len(func_tag))]):
                        has_partial = True
                        break
                
                if not has_partial:
                    yield {"type": "token", "content": buffer}
                    buffer = ""

        # Yield any remaining normal text if no tool call was triggered
        if buffer and not is_tool_call:
            yield {"type": "token", "content": buffer}

        # Handle tool calling execution if detected
        if is_tool_call and tool_call_text:
            # Normalize tool_call_text if it is in the format: function=name>args
            if "function=" in tool_call_text and not tool_call_text.startswith("<tool_call:"):
                match = re.search(r"function=([a-zA-Z0-9_]+)>(.*)", tool_call_text, re.DOTALL)
                if match:
                    name = match.group(1).strip()
                    args = match.group(2).strip()
                    if args.endswith("</tool_call>"):
                        args = args[:-12].strip()
                    tool_call_text = f"<tool_call:{name}>{args}</tool_call>"

            yield from self._handle_tool_call(tool_call_text, query, provider)

    def _handle_tool_call(self, tool_call_text: str, query: str, provider: str = "auto") -> Generator[dict, None, None]:
        # Parse XML tags
        match = re.match(r"<tool_call:([^>]+)>(.*?)</tool_call>", tool_call_text, re.DOTALL)
        if not match:
            match = re.search(r"<tool_call:([^>]+)>(.*)", tool_call_text, re.DOTALL)

        if match:
            tool_name = match.group(1).strip()
            args_str = match.group(2).strip()
            if args_str.endswith("</tool_call>"):
                args_str = args_str[:-12].strip()

            # Helper for robust parsing of unescaped literal newlines inside JSON values
            def parse_json_args(a_str: str) -> dict:
                try:
                    return json.loads(a_str)
                except Exception:
                    try:
                        args_dict = {}
                        for key in ["code", "url", "search_query"]:
                            pattern = r'"' + key + r'"\s*:\s*"(.*?)"(?=\s*(,|\s*\}))'
                            m = re.search(pattern, a_str, re.DOTALL)
                            if m:
                                val = m.group(1)
                                val = val.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                                args_dict[key] = val
                        return args_dict
                    except Exception as parse_err:
                        logger.error(f"Failed to manually parse JSON args: {parse_err}")
                        return {}

            args = parse_json_args(args_str)

            if tool_name == "execute_python_code":
                code = args.get("code", "")
                if code:
                    yield {"type": "terminal", "content": f"🤖 Code Interpreter: Đang chạy mã Python phân tích...\n\n```python\n{code}\n```"}
                    
                    from app.application.code_interpreter_usecase import CodeInterpreterUseCase
                    use_case = CodeInterpreterUseCase()
                    result = use_case.execute(code)

                    if result["success"]:
                        yield {"type": "terminal", "content": f"✅ Thực thi thành công!\n\nStdout:\n{result['stdout']}"}
                        # Yield any generated chart paths
                        for chart in result["charts"]:
                            yield {"type": "code_chart", "content": chart}

                        # Invoke LLM to summarize outputs
                        context_ext = f"Python Code:\n{code}\n\nExecution Output:\n{result['stdout']}"
                        yield {"type": "token", "content": "\n\n### Kết quả phân tích:\n"}
                        for tok in self.llm_service.generate_answer(
                            context=context_ext,
                            query=f"Explain this python code execution output and results to the user: {query}",
                            provider=provider
                        ):
                            yield {"type": "token", "content": tok}
                    else:
                        yield {"type": "terminal", "content": f"❌ Thực thi thất bại!\n\nStderr:\n{result['stderr']}"}
                        yield {"type": "token", "content": f"\n\n⚠️ Lỗi chạy code: {result['stderr']}"}

            elif tool_name == "web_automation_download":
                url = args.get("url", "")
                search_query = args.get("search_query", "")

                if url:
                    yield {"type": "terminal", "content": f"🤖 Browser Agent: Khởi chạy Playwright tải tài liệu từ {url}..."}
                    yield {"type": "terminal", "content": f"🌐 Tìm kiếm liên kết tải xuống khớp với: '{search_query}'"}

                    from app.infrastructure.browser_agent import BrowserAgent
                    import asyncio

                    agent = BrowserAgent()
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    filepath = loop.run_until_complete(agent.execute_download(url, search_query))
                    loop.close()

                    yield {"type": "terminal", "content": f"💾 Đã tải thành công tài liệu về máy chủ: {filepath}"}
                    yield {"type": "terminal", "content": "⚙️ Tiến hành phân mảnh (chunking) và tạo embeddings lưu vào CSDL..."}

                    # Index new content
                    from app.application.upload_usecase import UploadUseCase
                    upload_uc = UploadUseCase(vector_store=self.vector_store)
                    with open(filepath, "rb") as f:
                        file_content = f.read()
                    
                    filename = os.path.basename(filepath)
                    upload_res = upload_uc.execute(file_content, filename)

                    yield {"type": "terminal", "content": f"✅ Đã lập chỉ mục xong {upload_res['total_chunks']} phân đoạn tài liệu!"}

                    # Re-run query on updated context
                    yield {"type": "token", "content": "\n\n### Kết quả phân tích từ tài liệu mới:\n"}
                    new_results = self.vector_store.search_similar(query)
                    new_context = "\n\n".join([item["text"] for item in new_results])
                    for tok in self.llm_service.generate_answer(context=new_context, query=query, provider=provider):
                        yield {"type": "token", "content": tok}
