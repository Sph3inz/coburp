import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, cast
import json
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from fast_graphrag import GraphRAG
from fast_graphrag._llm._base import BaseLLMService, BaseEmbeddingService
from fast_graphrag._models import BaseModelAlias
import os
import shutil
import datetime
import aiofiles
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    PruningContentFilter,
    DefaultMarkdownGenerator
)
import argparse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

T_model = TypeVar('T_model')

class CustomAnswer(BaseModel):
    answer: str

@dataclass
class GeminiEmbeddingService(BaseEmbeddingService):
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
        # Offload the synchronous encoding to a background thread
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(texts, convert_to_numpy=True)
        )
        return embeddings.astype(np.float32)

@dataclass
class GeminiLLMService(BaseLLMService):
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

    def _clean_json_response(self, text: str) -> str:
        """Cleans markdown formatting from the response text."""
        # If wrapped in triple backticks, extract the inner content
        if text.startswith("```") and text.endswith("```"):
            lines = text.split("\n")
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])
        # Remove any leading specifier like "json\n"
        if text.startswith("json\n"):
            text = text[5:]
        return text.strip()

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
        """Send message to Gemini."""
        temperature = temperature or self.temperature
        messages: List[dict[str, str]] = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        # When a response model is provided, build the schema instruction from its JSON schema.
        if response_model:
            model_class = (response_model.Model if issubclass(response_model, BaseModelAlias)
                           else response_model)
            schema = model_class.model_json_schema()
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
        else:
            schema_instruction = (
                "Your response must be a valid JSON object with these fields:\n"
                "{\n"
                '    "entities": [\n'
                '        {"name": "string", "type": "string", "desc": "string"}\n'
                '    ],\n'
                '    "relationships": [\n'
                '        {"source": "string", "target": "string", "desc": "string"}\n'
                '    ],\n'
                '    "other_relationships": []\n'
                "}"
            )
            messages.insert(0, {"role": "system", "content": schema_instruction})

        combined_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

        try:
            response = await self._gemini_model.generate_content_async(
                contents=combined_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    candidate_count=1,
                    top_p=0.8,
                    top_k=40,
                )
            )

            if not response or not response.text:
                raise Exception("Empty response from Gemini")

            # Clean response text using _clean_json_response method
            response_text = self._clean_json_response(response.text)

            if response_model:
                if issubclass(response_model, BaseModelAlias):
                    llm_response = response_model.Model.model_validate_json(response_text)
                else:
                    llm_response = response_model.model_validate_json(response_text)
            else:
                llm_response = response_text

            messages.append({
                "role": "assistant",
                "content": response_text
            })

            if response_model and issubclass(response_model, BaseModelAlias):
                llm_response = cast(T_model, cast(BaseModelAlias.Model, llm_response).to_dataclass(llm_response))

            return llm_response, messages

        except Exception as e:
            logger.error(f"Error in send_message: {str(e)}")
            raise

async def crawl_url(url: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Crawls the given URL and returns content and metadata if successful."""
    config = BrowserConfig(
        headless=True,
        verbose=False,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport_width=1920,
        viewport_height=1080
    )
    
    content_filter = PruningContentFilter(threshold=0.5, min_word_threshold=50)
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=content_filter,
        options={
            "heading_style": "atx",
            "bullet_char": "*",
            "code_block_style": "fenced",
            "preserve_emphasis": True,
            "preserve_tables": True,
            "preserve_links": False,
            "preserve_images": False,
            "escape_html": False,
            "wrap_width": 100,
            "ignore_links": True,
            "ignore_images": True,
            "skip_internal_links": True
        }
    )
    run_config = CrawlerRunConfig(
        word_count_threshold=10,
        excluded_tags=['form', 'header', 'footer', 'nav', 'script', 'style', 'noscript'],
        exclude_external_links=True,
        remove_overlay_elements=True,
        process_iframes=True,
        cache_mode=CacheMode.BYPASS,
        js_code="window.scrollTo(0, document.body.scrollHeight);",
        wait_until="networkidle",
        page_timeout=60000,
        verbose=False,
        markdown_generator=markdown_generator
    )
    print(f"[Crawling] {url}")
    try:
        async with AsyncWebCrawler(config=config) as crawler:
            result = await crawler.arun(url=url, config=run_config)
            if result.success:
                content = None
                if result.markdown_v2 and result.markdown_v2.raw_markdown and result.markdown_v2.raw_markdown.strip():
                    content = result.markdown_v2.raw_markdown
                elif result.markdown and ((isinstance(result.markdown, str) and result.markdown.strip()) or
                                           (hasattr(result.markdown, 'raw_markdown') and result.markdown.raw_markdown.strip())):
                    content = result.markdown if isinstance(result.markdown, str) else result.markdown.raw_markdown
                elif result.fit_markdown and result.fit_markdown.strip():
                    content = result.fit_markdown
                elif result.cleaned_html and result.cleaned_html.strip():
                    content = f"# {result.metadata.get('title', 'Untitled Page')}\n\n{result.cleaned_html}"
                if content and content.strip():
                    metadata = {
                        "url": url,
                        "title": result.metadata.get("title") if result.metadata else None,
                        "crawl_time": str(datetime.datetime.now()),
                        "success": True
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
    except Exception as e:
        print(f"[Error] Exception while crawling {url}: {e}")
        return None

async def crawl_writeups(grag, writeups_path: str = "writeups.json") -> None:
    """Reads writeups.json, crawls all links, and inserts their content into GraphRAG."""
    BATCH_SIZE = 2  # Process only 2 links at a time
    MAX_CONCURRENT_CRAWLS = 1  # Only 1 concurrent crawl
    DELAY_BETWEEN_CRAWLS = 3.0  # 3 seconds delay between individual crawls
    DELAY_BETWEEN_BATCHES = 10.0  # 10 seconds delay between batches
    MAX_RETRIES = 3  # Maximum number of retries for rate limit errors
    BASE_RETRY_DELAY = 5.0  # Base delay for exponential backoff

    if not os.path.exists(writeups_path):
        print(f"[Error] '{writeups_path}' not found.")
        return
    try:
        async with aiofiles.open(writeups_path, "r", encoding="utf-8") as f:
            file_content = await f.read()
    except Exception as e:
        print(f"[Error] Could not open '{writeups_path}': {e}")
        return

    if not file_content.strip():
        print(f"[Warn] '{writeups_path}' is empty.")
        return

    try:
        json_data = json.loads(file_content)
        if not isinstance(json_data, dict) or "data" not in json_data:
            print("[Error] Expected JSON with 'data' field in writeups.json")
            return
        data = json_data["data"]
        if not isinstance(data, list):
            print("[Error] Expected 'data' field to be a list in writeups.json")
            return
    except json.JSONDecodeError as e:
        print(f"[Error] Invalid JSON in '{writeups_path}': {e}")
        return

    links = []
    for entry in data:
        if isinstance(entry, dict) and "Links" in entry and isinstance(entry["Links"], list):
            for link_item in entry["Links"]:
                if isinstance(link_item, dict) and "Link" in link_item:
                    url = link_item["Link"]
                    if url and isinstance(url, str) and url.strip():
                        links.append(url.strip())
    if not links:
        print("[Info] No links found in writeups.json")
        return

    total_links = len(links)
    print(f"[Info] Found {total_links} links in writeups.json")
    
    # Split links into batches
    batches = [links[i:i + BATCH_SIZE] for i in range(0, total_links, BATCH_SIZE)]
    total_batches = len(batches)

    # Process each batch
    for batch_num, batch in enumerate(batches, 1):
        print(f"\n[Batch {batch_num}/{total_batches}] Processing {len(batch)} links...")
        
        for url in batch:
            retries = 0
            while retries < MAX_RETRIES:
                try:
                    result = await crawl_url(url)
                    if result:
                        content, metadata = result
                        try:
                            await grag.async_insert(content, metadata=metadata)
                            print(f"[Inserted] Content from {metadata['url']}")
                            # Successful processing, break retry loop
                            break
                        except Exception as e:
                            if "429" in str(e):
                                retries += 1
                                retry_delay = BASE_RETRY_DELAY * (2 ** retries)
                                print(f"[Rate Limited] Attempt {retries}/{MAX_RETRIES}. Waiting {retry_delay} seconds...")
                                await asyncio.sleep(retry_delay)
                            else:
                                print(f"[Error] Failed to insert content from {metadata['url']}: {str(e)}")
                                break
                        finally:
                            # Add delay after each successful processing
                            await asyncio.sleep(DELAY_BETWEEN_CRAWLS)
                except Exception as e:
                    if "429" in str(e):
                        retries += 1
                        retry_delay = BASE_RETRY_DELAY * (2 ** retries)
                        print(f"[Rate Limited] Attempt {retries}/{MAX_RETRIES}. Waiting {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                    else:
                        print(f"[Error] Failed to process {url}: {str(e)}")
                        break
        
        # Small delay between batches
        if batch_num < total_batches:
            print(f"\n[Info] Waiting {DELAY_BETWEEN_BATCHES} seconds before processing batch {batch_num + 1}...")
            await asyncio.sleep(DELAY_BETWEEN_BATCHES)

async def main(args):
    """Test GraphRAG with a simple example."""
    # Reset the working directory for testing to avoid metadata loading errors
    working_dir = "./graphdata_test"
    if args.reset_index and os.path.exists(working_dir):
        print(f"Resetting working directory: {working_dir}")
        shutil.rmtree(working_dir)
    
    GEMINI_API_KEY = "AIzaSyDyjlf1FL2xxcDpNFA1doh7Aw8biNMqZTs"
    
    # Initialize services
    embedding_service = GeminiEmbeddingService(
        api_key=GEMINI_API_KEY,
        model="all-MiniLM-L6-v2"
    )
    
    llm_service = GeminiLLMService(
        model="gemini-2.0-flash-lite-preview-02-05",
        api_key=GEMINI_API_KEY
    )
    
    # Initialize GraphRAG with comprehensive universal entity types
    grag = GraphRAG(
        working_dir="./graphdata_test",
        domain="Universal Domain",
        example_queries=[
            "Summarize the key concepts in the provided text.",
            "What are the primary entities mentioned in the text?"
        ],
        entity_types=[
            "named", "generic", "numerical", "date", "location", 
            "organization", "person", "concept", "event", "miscellaneous",
            "product", "technology", "industry", "software", "hardware", 
            "protocol", "risk", "vulnerability", "threat"
        ],
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service
        )
    )
    
    if args.test:
        # Optional test document insertion and query
        test_doc = """
        # Web Security Best Practices
    
        In today's digital landscape, maintaining robust web security is essential for safeguarding your data...
        """
        print("Inserting test document...")
        await grag.async_insert(test_doc)
    
        print("\nTesting query...")
        query = "Provide a detailed summary of the document, including key practices and insights."
        try:
            result = await grag.async_query(query)
            print(f"\nQuery: {query}")
            try:
                parsed = CustomAnswer.model_validate_json(result.response)
                print(f"Response: {parsed.answer}")
            except Exception as e:
                print(f"Response (raw): {result.response}")
        except Exception as e:
            print(f"Query test failed: {str(e)}")

    # NEW: Now crawl links from writeups.json and insert their content into GraphRAG
    print("\nStarting crawl of links from writeups.json...")
    await crawl_writeups(grag, writeups_path=args.writeups_path)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Test GraphRAG with a simple example.")
        parser.add_argument(
            "--reset-index",
            action="store_true",
            help="Clear existing GraphRAG index to ensure embedding consistency."
        )
        parser.add_argument(
            "--writeups-path",
            type=str,
            default=r"D:\pentestbro\burp-extension\coburp\writeups.json",
            help="Path to the writeups.json file"
        )
        parser.add_argument(
            "--test",
            action="store_true",
            help="Run the test document insertion and query"
        )
        args = parser.parse_args()

        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nTest interrupted.")
    except Exception as e:
        print(f"Error during test: {str(e)}") 