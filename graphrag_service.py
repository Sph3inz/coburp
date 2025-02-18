from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import asyncio
from fast_graphrag import GraphRAG
from dataclasses import dataclass, field
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models for request/response
class InsertRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str
    references: List[str]

# Initialize services and GraphRAG
@dataclass
class GeminiEmbeddingService:
    """Embedding service using sentence-transformers."""
    embedding_dim: int = field(default=768)
    max_elements_per_request: int = field(default=32)
    model: Optional[str] = field(default="all-MiniLM-L6-v2")
    api_key: Optional[str] = field(default=None)
    _model: Optional[SentenceTransformer] = field(default=None, init=False)

    def __post_init__(self):
        self._model = SentenceTransformer(self.model)
        self.embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Initialized SentenceTransformer with model {self.model}")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, convert_to_numpy=True)
        )
        return embeddings.astype(np.float32)

@dataclass
class GeminiLLMService:
    """Gemini implementation for LLM services."""
    model: Optional[str] = field(default="gemini-2.0-flash-lite-preview-02-05")
    api_key: Optional[str] = field(default=None)
    temperature: float = field(default=0.7)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    _gemini_model: Any = field(default=None, init=False)

    def __post_init__(self):
        if not self.model:
            raise ValueError("Model name must be provided.")
        if self.api_key:
            genai.configure(api_key=self.api_key)
        self._gemini_model = genai.GenerativeModel(self.model)

    async def send_message(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        combined_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        response = await self._gemini_model.generate_content_async(
            contents=combined_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                candidate_count=1,
                top_p=0.8,
                top_k=40,
            )
        )

        if not response or not response.text:
            raise Exception("Empty response from Gemini")

        return response.text

# Initialize FastAPI app
app = FastAPI(title="GraphRAG Service")

# Global GraphRAG instance
grag: Optional[GraphRAG] = None

@app.on_event("startup")
async def startup_event():
    global grag
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    # Initialize services
    embedding_service = GeminiEmbeddingService(api_key=api_key)
    llm_service = GeminiLLMService(api_key=api_key)

    # Initialize GraphRAG
    grag = GraphRAG(
        working_dir="./graphdata_test",
        domain="Web Security Domain",
        example_queries=[
            "Analyze the security implications of the web traffic.",
            "What vulnerabilities are present in the traffic?"
        ],
        entity_types=[
            "request", "response", "header", "parameter", "endpoint",
            "vulnerability", "authentication", "authorization", "injection",
            "xss", "sqli", "csrf", "idor", "sensitive_data", "misconfiguration"
        ],
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service
        )
    )

@app.post("/insert")
async def insert_content(request: InsertRequest):
    if not grag:
        raise HTTPException(status_code=500, detail="GraphRAG not initialized")
    try:
        await grag.async_insert(request.content, metadata=request.metadata)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_graphrag(request: QueryRequest):
    if not grag:
        raise HTTPException(status_code=500, detail="GraphRAG not initialized")
    try:
        result = await grag.async_query(request.query)
        return QueryResponse(
            response=result.response,
            references=result.references
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000) 