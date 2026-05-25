from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    provider: str = "auto"
    notebook_id: str = "default"

class QueryResponse(BaseModel):
    answer: str

class SynthesizeNotesRequest(BaseModel):
    notes: list[str]
    action: str
    provider: str = "auto"

class StudioGenerateRequest(BaseModel):
    content_type: str
    provider: str = "auto"
    notebook_id: str = "default"

class PodcastGenerateRequest(BaseModel):
    provider: str = "auto"
    notebook_id: str = "default"
    custom_instructions: str = ""

class SlideGenerateRequest(BaseModel):
    provider: str = "auto"
    notebook_id: str = "default"
    num_slides: int = 10
    chat_context: str | None = None

