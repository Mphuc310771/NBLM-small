import json
import logging
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.domain.models import QueryRequest, SynthesizeNotesRequest, StudioGenerateRequest, PodcastGenerateRequest, SlideGenerateRequest
from app.application.podcast_usecase import PodcastUseCase
from app.application.synthesis_usecase import SynthesisUseCase
from app.application.studio_usecase import StudioUseCase
from app.application.slide_usecase import SlideGeneratorUseCase
from app.infrastructure.vector_store import ChromaDBStore
from app.infrastructure.groq_adapter import GroqAdapter
from app.infrastructure.gemini_adapter import GeminiAdapter
from app.infrastructure.openrouter_adapter import OpenRouterAdapter
from app.infrastructure.sambanova_adapter import SambaNovaAdapter
from app.infrastructure.mistral_adapter import MistralAdapter
from app.infrastructure.fallback_llm import FallbackLLMService
from app.application.rag_usecase import RAGUseCase
from app.application.upload_usecase import UploadUseCase
from app.application.quiz_usecase import QuizGeneratorUseCase
from app.application.summary_usecase import DocumentSummaryUseCase
from app.application.graph_usecase import GraphUseCase
from app.application.delete_usecase import DeleteUseCase

logger = logging.getLogger(__name__)
router = APIRouter()

# ====================================================================
# Dependency Injection — Singleton Pattern
# Instantiate dependencies once at module level so the embedding model
# and ChromaDB client are reused across all requests.
# ====================================================================
vector_store = ChromaDBStore()
groq_provider = GroqAdapter()
gemini_provider = GeminiAdapter()
openrouter_provider = OpenRouterAdapter()
sambanova_provider = SambaNovaAdapter()
mistral_provider = MistralAdapter()
llm_service = FallbackLLMService(
    groq_adapter=groq_provider, 
    gemini_adapter=gemini_provider, 
    openrouter_adapter=openrouter_provider,
    sambanova_adapter=sambanova_provider,
    mistral_adapter=mistral_provider
)


def get_rag_use_case() -> RAGUseCase:
    return RAGUseCase(vector_store=vector_store, llm_service=llm_service)


def get_upload_use_case() -> UploadUseCase:
    return UploadUseCase(vector_store=vector_store)


def get_quiz_use_case() -> QuizGeneratorUseCase:
    return QuizGeneratorUseCase(vector_store=vector_store, llm_service=llm_service)


def get_summary_use_case() -> DocumentSummaryUseCase:
    return DocumentSummaryUseCase(vector_store=vector_store, llm_service=llm_service)


def get_graph_use_case() -> GraphUseCase:
    return GraphUseCase(vector_store=vector_store, llm_service=llm_service)


def get_delete_use_case() -> DeleteUseCase:
    return DeleteUseCase(vector_store=vector_store)


def get_podcast_use_case() -> PodcastUseCase:
    return PodcastUseCase(vector_store=vector_store, llm_service=llm_service)


def get_slide_use_case() -> SlideGeneratorUseCase:
    return SlideGeneratorUseCase(vector_store=vector_store, llm_service=llm_service)


def get_synthesis_use_case() -> SynthesisUseCase:
    return SynthesisUseCase(llm_service=llm_service)


def get_studio_use_case() -> StudioUseCase:
    return StudioUseCase(vector_store=vector_store, llm_service=llm_service)


@router.post("/ask")
def ask_question(request: QueryRequest, use_case: RAGUseCase = Depends(get_rag_use_case)):
    """
    Endpoint to process a user query using Retrieval-Augmented Generation.
    Returns a stream of JSON messages (either citation metadata or response tokens)
    formatted as a Server-Sent Event (SSE) stream.
    """
    def event_generator():
        try:
            for chunk in use_case.execute(request.query, provider=request.provider, notebook_id=request.notebook_id):
                # Format chunk as a standard Server-Sent Event (SSE) message
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error(f"Error in ask event stream: {e}", exc_info=True)
            err_msg = {"type": "error", "content": f"Internal server error: {str(e)}"}
            yield f"data: {json.dumps(err_msg, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/upload")
async def upload_document(notebook_id: str = "default", file: UploadFile = File(...), use_case: UploadUseCase = Depends(get_upload_use_case)):
    """
    Endpoint to upload a document (TXT or PDF), extract text,
    chunk it, and store embeddings in ChromaDB.
    """
    # Validate file type
    if not file.filename.lower().endswith(('.txt', '.pdf')):
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are supported.")

    try:
        await file.seek(0)
        file_content = await file.read()
        result = use_case.execute(file_content=file_content, filename=file.filename, notebook_id=notebook_id)
        return result
    except Exception as e:
        logger.error(f"Error processing /upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/generate-quiz")
def generate_quiz(notebook_id: str = "default", chat_context: str | None = None, use_case: QuizGeneratorUseCase = Depends(get_quiz_use_case)):
    """
    Endpoint to generate a 3-question multiple choice quiz from a random document chunk.
    """
    result = use_case.execute(notebook_id=notebook_id, chat_context=chat_context)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result.get("questions")


@router.get("/summarize")
def summarize_document(notebook_id: str = "default", use_case: DocumentSummaryUseCase = Depends(get_summary_use_case)):
    """
    Endpoint to generate a concise summary from the uploaded documents.
    """
    result = use_case.execute(notebook_id=notebook_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("summary"))
    return result


@router.get("/generate-graph")
def generate_graph(notebook_id: str = "default", chat_context: str | None = None, use_case: GraphUseCase = Depends(get_graph_use_case)):
    """
    Endpoint to generate a Mermaid.js visual mindmap from the uploaded documents.
    """
    result = use_case.execute(notebook_id=notebook_id, chat_context=chat_context)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("graph"))
    return result


@router.delete("/delete-document")
def delete_document(filename: str, notebook_id: str = "default", use_case: DeleteUseCase = Depends(get_delete_use_case)):
    """
    Endpoint to delete all chunks of a document from ChromaDB.
    """
    result = use_case.execute(filename=filename, notebook_id=notebook_id)
    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@router.get("/uploaded-files")
def get_uploaded_files(notebook_id: str = "default"):
    """
    Get the list of all unique document names currently in ChromaDB (excluding screen_capture) for this notebook.
    """
    try:
        results = vector_store.collection.get(where={"notebook_id": notebook_id}, include=["metadatas"])
        metadatas = results.get("metadatas", []) or []
        
        source_counts = {}
        for meta in metadatas:
            if meta and "source" in meta:
                src = meta["source"]
                if src != "screen_capture":
                    source_counts[src] = source_counts.get(src, 0) + 1
                
        file_items = []
        for src, chunks_count in sorted(source_counts.items()):
            file_items.append({
                "name": src,
                "status": f"✓ {chunks_count} chunks",
                "type": "success"
               })
        return file_items
    except Exception as e:
        logger.error(f"Error fetching uploaded files list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-podcast")
async def generate_podcast(request: PodcastGenerateRequest, use_case: PodcastUseCase = Depends(get_podcast_use_case)):
    """
    Generate an audio podcast briefing dialogue script from current documents.
    """
    result = await use_case.execute(
        provider=request.provider, 
        notebook_id=request.notebook_id, 
        custom_instructions=request.custom_instructions
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


@router.post("/generate-slides")
def generate_slides(request: SlideGenerateRequest, use_case: SlideGeneratorUseCase = Depends(get_slide_use_case)):
    """
    Generate interactive presentation slide content (JSON deck) from documents.
    """
    result = use_case.execute(
        provider=request.provider,
        notebook_id=request.notebook_id,
        num_slides=request.num_slides,
        chat_context=request.chat_context
    )
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result



@router.post("/synthesize-notes")
def synthesize_notes(request: SynthesizeNotesRequest, use_case: SynthesisUseCase = Depends(get_synthesis_use_case)):
    """
    Synthesize user notes into study guides, essays, summaries, or contradiction logs.
    """
    result = use_case.execute(notes=request.notes, action=request.action, provider=request.provider)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("result"))
    return result


@router.post("/studio/generate")
def studio_generate(request: StudioGenerateRequest, use_case: StudioUseCase = Depends(get_studio_use_case)):
    """
    Generate Studio content (flashcards, FAQ, timeline, study guide, briefing) from documents.
    """
    result = use_case.execute(content_type=request.content_type, provider=request.provider, notebook_id=request.notebook_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result
