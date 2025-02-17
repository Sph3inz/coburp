# DEPENDENCIES: pip install fast-graphrag google-generativeai
# Gemini: https://ai.google.dev/tutorials/python_quickstart
# Run this example: python gemini_llm.py

import asyncio
import re
import time
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, cast, Literal, TypeVar, Callable
from functools import wraps
import logging
import instructor
import json
import signal
import sys
from pathlib import Path
import datetime
import aiofiles  # NEW: For asynchronous file I/O
import os  # NEW: For checking and deleting index folder
import shutil  # NEW: For deleting the index folder

import numpy as np
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from sentence_transformers import SentenceTransformer

import google.generativeai as genai
from google.generativeai.types import content_types
from google.generativeai.types import model_types

from fast_graphrag._llm._base import BaseLLMService, BaseEmbeddingService
from fast_graphrag._models import BaseModelAlias
from fast_graphrag._models import _json_schema_slim

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

T_model = TypeVar("T_model")

# NEW IMPORTS for Crawl4AI:
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

RETRY_TIME_REGEX = re.compile(r'Retry after (\d+) seconds')  # NEW: Pre-compiled regex for retry extraction

def throttle_async_func_call(
    max_concurrent: int = 2048, stagger_time: Optional[float] = None, waiting_time: float = 0.001
):
    _wrappedFn = TypeVar("_wrappedFn", bound=Callable[..., Any])

    def decorator(func: _wrappedFn) -> _wrappedFn:
        __current_exes = 0
        __current_queued = 0

        @wraps(func)
        async def wait_func(*args: Any, **kwargs: Any) -> Any:
            nonlocal __current_exes, __current_queued
            while __current_exes >= max_concurrent:
                await asyncio.sleep(waiting_time)

            __current_exes += 1
            result = await func(*args, **kwargs)
            __current_exes -= 1
            return result

        return wait_func  # type: ignore

    return decorator

class LLMServiceNoResponseError(Exception):
    """Raised when the LLM service returns no response."""
    pass

@dataclass
class GeminiEmbeddingService(BaseEmbeddingService):
    """Embedding service using sentence-transformers."""
    # Custom configuration for embedding model
    embedding_dim: int = field(default=768)  # Embedding dimension
    max_elements_per_request: int = field(default=32)  # Batch size
    model: Optional[str] = field(default="all-mpnet-base-v2")  # Model name
    api_key: Optional[str] = field(default=None)  # Not used for sentence-transformers
    _model: Optional[SentenceTransformer] = field(default=None, init=False)
    device: Optional[str] = field(default=None)  # NEW: Device setting for model (e.g. "cuda" or "cpu")

    def __post_init__(self):
        """Initialize the sentence-transformers model and update embedding dimension."""
        if self.device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        self._model = SentenceTransformer(self.model, device=self.device)
        # Update the embedding dimension based on the model's output
        self.embedding_dim = self._model.get_sentence_embedding_dimension()
        logger.debug(
            f"Initialized SentenceTransformer with model {self.model} on device {self.device} "
            f"with embedding_dim {self.embedding_dim}"
        )

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Generates embeddings for a list of texts using batched processing.
        
        Args:
            texts: List of strings to embed.
            model: Optional model override (not used).
        Returns:
            Numpy array of embeddings.
        """
        # Custom batching implementation
        batched_texts = [
            texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
            for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
        ]

        loop = asyncio.get_running_loop()
        # Offload CPU-bound embedding computation to the default executor, concurrently for each batch
        batch_tasks = [
            loop.run_in_executor(None, self._embedding_request_sync, batch, model)
            for batch in batched_texts
        ]
        response = await asyncio.gather(*batch_tasks)

        # Flatten the batched responses and convert to numpy array
        embeddings = np.vstack(response)
        logger.debug(f"Received embedding response: {len(embeddings)} embeddings")

        return embeddings

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((TimeoutError, Exception)),
    )
    def _embedding_request_sync(self, input_texts: List[str], model: Optional[str]) -> np.ndarray:
        """Synchronously get embeddings for a batch of texts.

        Args:
            input_texts (List[str]): Batch of texts to embed.
            model (str): Model name (not used).

        Returns:
            np.ndarray: Array of embeddings for the batch.
        """
        try:
            embeddings = self._model.encode(input_texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error in embedding request: {str(e)}")
            raise

@dataclass
class GeminiLLMService(BaseLLMService):
    """Gemini implementation for LLM services."""
    model: Optional[str] = field(default="gemini-2.0-flash-lite-preview-02-05")  # Updated model name
    api_key: Optional[str] = field(default=None)
    
    # Other fields with defaults
    max_retries: int = field(default=3)
    retry_delay: float = field(default=2.0)
    rate_limit_max_retries: int = field(default=5)
    mode: instructor.Mode = field(default=instructor.Mode.JSON)
    temperature: float = field(default=0.6)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    llm_calls_count: int = field(default=0, init=False)
    _gemini_model: Any = field(default=None, init=False)
    
    def __post_init__(self):
        """Initialize after dataclass initialization."""
        if self.model is None:
            raise ValueError("Model name must be provided.")
            
        if self.api_key:
            genai.configure(api_key=self.api_key)
            
        # Configure safety settings to be more permissive
        safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "block_none",
            "HARM_CATEGORY_HATE_SPEECH": "block_none",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "block_none",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "block_none",
        }
        
        self._gemini_model = genai.GenerativeModel(
            self.model,
            safety_settings=safety_settings
        )

    def _extract_retry_time(self, error_message: str) -> float:
        # Use pre-compiled regex
        match = RETRY_TIME_REGEX.search(str(error_message))
        if match:
            return float(match.group(1))
        # Increased default retry time for rate limits
        return 5.0

    def _clean_json_response(self, text: str) -> str:
        """Clean the response text to extract valid JSON.
        
        Args:
            text: Raw response text that might contain markdown or other formatting
            
        Returns:
            Clean JSON string
        """
        # Remove markdown code blocks if present
        if text.startswith("```") and text.endswith("```"):
            # Extract content between code blocks
            lines = text.split("\n")
            if len(lines) > 2:
                # Remove first and last lines (``` markers)
                text = "\n".join(lines[1:-1])
            
        # Remove any "json" or other language specifiers
        if text.startswith("json\n"):
            text = text[5:]
            
        # Remove any trailing whitespace
        text = text.strip()
        
        return text

    @throttle_async_func_call(
        max_concurrent=20,  # Reduced from 50
        stagger_time=0.2,  # Increased from 0.1
        waiting_time=0.005  # Increased from 0.001
    )
    async def send_message(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        history_messages: Optional[List[dict[str, str]]] = None,
        response_model: Optional[Type[T_model]] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> Tuple[T_model, List[dict[str, str]]]:
        """Sends a message to the Gemini LLM and handles the response."""
        temperature = temperature or self.temperature
        retries = 0
        rate_limit_retries = 0
        
        logger.debug(f"Sending message with prompt: {prompt}")
        model = model or self.model
        if model is None:
            raise ValueError("Model name must be provided.")
        
        messages: List[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            logger.debug(f"Added system prompt: {system_prompt}")

        if history_messages:
            messages.extend(history_messages)
            logger.debug(f"Added history messages: {history_messages}")

        messages.append({"role": "user", "content": prompt})

        # Add format instruction to the prompt if response_model exists
        if response_model:
            model_class = (response_model.Model 
                        if issubclass(response_model, BaseModelAlias)
                        else response_model)
            schema = model_class.model_json_schema()
            
            # Enhanced schema instruction
            schema_instruction = (
                "IMPORTANT: Your response must be a valid JSON object. DO NOT wrap it in markdown code blocks.\n\n"
                "Follow this schema exactly:\n"
                f"{schema}\n\n"
                "Requirements:\n"
                "1. Response must be pure JSON - no markdown, no code blocks\n"
                "2. All required fields must be included\n"
                "3. Values must match the specified types\n"
                "4. Arrays can be empty [] but must be included\n"
            )
            messages.insert(0, {"role": "system", "content": schema_instruction})

        # Combine messages into a single prompt for Gemini
        combined_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        while True:
            try:
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    candidate_count=1,
                    top_p=0.8,
                    top_k=40,
                )

                # Send request to Gemini
                response = await self._gemini_model.generate_content_async(
                    contents=combined_prompt,
                    generation_config=generation_config,
                )

                # Validate response
                if not response or not response.text:
                    logger.error("Empty response from Gemini")
                    raise LLMServiceNoResponseError("Empty response from Gemini")

                # Clean and extract JSON from response
                response_text = self._clean_json_response(response.text)

                # Parse response using response_model if provided
                try:
                    if response_model:
                        if issubclass(response_model, BaseModelAlias):
                            llm_response = response_model.Model.model_validate_json(response_text)
                        else:
                            llm_response = response_model.model_validate_json(response_text)
                    else:
                        llm_response = response_text
                except ValidationError as e:
                    logger.error(f"JSON validation error: {str(e)}\nResponse text: {response_text}")
                    raise LLMServiceNoResponseError(f"Invalid JSON response: {str(e)}") from e

                self.llm_calls_count += 1

                if not llm_response:
                    logger.error("No response received from the language model.")
                    raise LLMServiceNoResponseError("No response received from the language model.")

                messages.append({
                    "role": "assistant",
                    "content": (llm_response.model_dump_json() 
                              if isinstance(llm_response, BaseModel) 
                              else str(llm_response)),
                })
                logger.debug(f"Received response: {llm_response}")

                if response_model and issubclass(response_model, BaseModelAlias):
                    llm_response = cast(T_model, cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response))

                return llm_response, messages

            except Exception as e:
                if "Rate limit exceeded" in str(e) or "429" in str(e):
                    if rate_limit_retries >= self.rate_limit_max_retries:
                        error_log = (
                            f"Rate limit max retries reached ({self.rate_limit_max_retries})|{str(e)}"
                        )
                        logger.error(error_log)
                        raise Exception(f"Rate limit exceeded after {self.rate_limit_max_retries} retries: {e}") from e
                    
                    retry_time = self._extract_retry_time(str(e))
                    rate_limit_retries += 1
                    error_log = f"Rate limit hit (attempt {rate_limit_retries})|{str(e)}"
                    logger.warning(error_log)
                    
                    # Exponential backoff for rate limits
                    await asyncio.sleep(retry_time * (2 ** rate_limit_retries))
                    continue
                
                if retries >= self.max_retries:
                    error_log = f"Max retries reached ({self.max_retries})|{str(e)}"
                    logger.error(error_log)
                    raise Exception(f"LLM API failed after {self.max_retries} retries: {e}") from e

                retries += 1
                wait_time = self.retry_delay * (2 ** retries)  # Exponential backoff
                error_log = f"Attempt {retries}|{str(e)}"
                logger.warning(error_log)

                await asyncio.sleep(wait_time)
                continue

# NEW: Function to crawl a single URL and return the content and metadata
async def crawl_url(url: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Crawls the given URL and returns content and metadata if successful."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=True
    )
    
    run_config = CrawlerRunConfig(
        word_count_threshold=10,
        excluded_tags=['form', 'header', 'footer', 'nav'],
        exclude_external_links=True,
        remove_overlay_elements=True,
        process_iframes=True,
        cache_mode=CacheMode.BYPASS,
    )
    
    print(f"\n[Crawling] {url}")
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        if result.success:
            content = None
            if result.markdown_v2:
                content = result.markdown_v2.raw_markdown
            elif result.markdown:
                content = result.markdown if isinstance(result.markdown, str) else result.markdown.raw_markdown
            elif result.cleaned_html:
                content = result.cleaned_html
            elif result.html:
                content = result.html
                
            if content:
                metadata = {
                    "url": url,
                    "title": result.metadata.get("title") if result.metadata else None,
                    "crawl_time": str(datetime.datetime.now()),
                }
                print(f"[Success] Crawled {url}")
                if result.metadata and result.metadata.get("title"):
                    print(f"[Info] Title: {result.metadata['title']}")
                return content, metadata
            else:
                print(f"[Warn] No usable content found for {url}")
                return None
        else:
            print(f"[Error] Failed to crawl {url}: {result.error_message}")
            return None

# NEW: Function to batch process links and insert into GraphRAG
async def process_link_batch(grag, urls: List[str], insert_queue: asyncio.Queue) -> None:
    """Process a batch of URLs and insert their content into GraphRAG."""
    contents_and_metadata = []
    
    # Crawl all URLs concurrently instead of sequentially
    crawl_tasks = [crawl_url(url) for url in urls]
    crawl_results = await asyncio.gather(*crawl_tasks, return_exceptions=True)
    
    for url, result in zip(urls, crawl_results):
        if isinstance(result, Exception):
            print(f"[Error] Failed to process {url}: {str(result)}")
        elif result:
            content, metadata = result
            contents_and_metadata.append((content, metadata))
    
    # Enqueue all successful crawls for insertion
    if contents_and_metadata:
        print(f"\n[Enqueuing] Batch of {len(contents_and_metadata)} documents for insertion...")
        for content, metadata in contents_and_metadata:
            await insert_queue.put({"content": content, "metadata": metadata})
    else:
        print("[Warn] No content to enqueue from this batch")

# NEW: Function to continuously crawl links in batches.
async def continuous_crawl_loop(grag, start_link_index: int = 0, insert_queue: asyncio.Queue = None) -> None:
    """Continuously reads 'writeups.json' and crawls links in batches.
    
    Args:
        grag: The GraphRAG instance.
        start_link_index: The starting link number (1-indexed). Links with a count
            lower than this value will be skipped.
    """
    processed_links = set()
    link_counter = 0   # New counter to track the number of new links discovered
    print("[Info] Starting continuous crawl loop. Press Ctrl+C to exit.")
    
    BATCH_SIZE = 4  # Process 4 links at a time
    
    while True:
        writeups_path = Path(r"D:\pentestbro\burp-extension\coburp\writeups.json")
        if not writeups_path.exists():
            print(f"[Warn] '{writeups_path}' not found. Waiting for the file to appear...")
            await asyncio.sleep(10)
            continue

        try:
            async with aiofiles.open(writeups_path, "r", encoding="utf-8") as f:
                file_content = await f.read()
        except Exception as e:
            print(f"[Error] Could not open 'writeups.json': {e}")
            await asyncio.sleep(10)
            continue

        if not file_content.strip():
            print("[Warn] 'writeups.json' is empty. Waiting for data...")
            await asyncio.sleep(10)
            continue

        try:
            json_data = json.loads(file_content)
            if not isinstance(json_data, dict) or "data" not in json_data:
                print("[Error] Expected JSON with 'data' field in writeups.json")
                await asyncio.sleep(10)
                continue

            data = json_data["data"]
            if not isinstance(data, list):
                print("[Error] Expected 'data' field to be a list in writeups.json")
                await asyncio.sleep(10)
                continue
        except json.JSONDecodeError as e:
            print(f"[Error] Invalid JSON in 'writeups.json': {e}")
            await asyncio.sleep(10)
            continue

        # After reading valid JSON data from writeups.json:
        # Collect new links, starting from the specified start_link_index
        new_links = []
        for entry in data:
            if isinstance(entry, dict) and "Links" in entry and isinstance(entry["Links"], list):
                for link_item in entry["Links"]:
                    if isinstance(link_item, dict) and "Link" in link_item:
                        url = link_item["Link"]
                        if url and isinstance(url, str) and url not in processed_links:
                            link_counter += 1
                            if link_counter >= start_link_index:
                                print(f"[Debug] Found new link {link_counter}: {url} (Title: {link_item.get('Title', 'No title')})")
                                new_links.append(url)
                            else:
                                print(f"[Skip] Skipping link {link_counter} (before start index {start_link_index}): {url}")
                            processed_links.add(url)

        if new_links:
            total_batches = (len(new_links) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"[Info] Found {len(new_links)} new link(s) to process in {total_batches} batch(es).")
            
            # Process links in batches
            for i in range(0, len(new_links), BATCH_SIZE):
                current_batch = i // BATCH_SIZE + 1
                batch = new_links[i:i + BATCH_SIZE]
                print(f"\n[Batch {current_batch}/{total_batches}] Processing {len(batch)} links...")
                await process_link_batch(grag, batch, insert_queue)
                
                # Reduced delay between batches
                if i + BATCH_SIZE < len(new_links):  # Only delay if there are more batches to process
                    print(f"[Info] Waiting 2 seconds before processing batch {current_batch + 1}...")
                    await asyncio.sleep(2)
        else:
            print("[Info] No new links found in 'writeups.json'.")
        
        # Wait before checking for new links
        print("[Info] Waiting 5 seconds before checking for new links...")
        await asyncio.sleep(5)

# NEW: Insertion worker for faster ingestion via a producer-consumer queue
async def insertion_worker(grag, insert_queue: asyncio.Queue):
    while True:
        document = await insert_queue.get()
        try:
            await grag.async_insert(document["content"], metadata=document["metadata"])
            print(f"[Inserted] Content from {document['metadata']['url']}")
        except Exception as e:
            print(f"[Error] Failed to insert content from {document['metadata']['url']}: {str(e)}")
        insert_queue.task_done()

if __name__ == "__main__":
    import argparse
    from fast_graphrag import GraphRAG

    # Parse command-line arguments for the starting link number and index reset flag.
    parser = argparse.ArgumentParser(
        description="Gemini LLM and GraphRAG continuous crawler with starting link number"
    )
    parser.add_argument(
        "--start-link",
        type=int,
        default=0,
        help="Start processing from this link number (1-indexed), e.g., 20 to start from the 20th link. Default is 0 (process all links)."
    )
    parser.add_argument(
        "--reset-index",
        action="store_true",
        help="Clear existing GraphRAG index to ensure embedding consistency."
    )
    args = parser.parse_args()
    
    # If the reset flag is used, clear the existing GraphRAG working directory
    if args.reset_index:
        if os.path.exists("./graphdata"):
            print("[Info] Resetting GraphRAG index by clearing the ./graphdata directory.")
            shutil.rmtree("./graphdata")
        else:
            print("[Info] No existing index found to reset.")

    # Sample API key (replace with your actual key)
    GEMINI_API_KEY = "AIzaSyAwjj3FkVASfvXfY71jla-fAhlpGF9Bdnc"
    
    # Initialize Gemini services with an existing API key.
    embedding_service = GeminiEmbeddingService(
        api_key=GEMINI_API_KEY,
        model="all-mpnet-base-v2"  # Use the sentence-transformers model
    )
    
    llm_service = GeminiLLMService(
        model="gemini-2.0-flash-lite-preview-02-05",  # Use the stable model
        api_key=GEMINI_API_KEY,
        max_retries=3,
        retry_delay=2.0,
        rate_limit_max_retries=5,
        temperature=0.7
    )
    
    # Initialize GraphRAG with a working directory in the current folder.
    grag = GraphRAG(
        working_dir="./graphdata",  # Data will be stored here
        domain="Web Penetration Testing and Bug Bounty",  # Updated domain
        example_queries="",
        entity_types=["Web", "BugBounty", "PenTest"],
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service 
        )
    )

    # Create a global insertion queue
    insert_queue = asyncio.Queue(maxsize=1000)

    async def main_async():
        # Start insertion worker tasks within the event loop
        NUM_WORKERS = 10  # Adjust number of workers as needed
        for _ in range(NUM_WORKERS):
            asyncio.create_task(insertion_worker(grag, insert_queue))

        await continuous_crawl_loop(grag, start_link_index=args.start_link, insert_queue=insert_queue)

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n[Exit] Exiting program. Data stored remains intact.")