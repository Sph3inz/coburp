import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
import numpy as np
import asyncio
import nest_asyncio
from google import genai
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

DEFAULT_RAG_DIR = "index_default"

# Configure working directory and environment variables
WORKING_DIR = os.environ.get("RAG_DIR", f"{DEFAULT_RAG_DIR}")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b")
EMBEDDING_MODEL = "text-embedding-004"  # Gemini embedding model
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
CEREBRAS_BASE_URL = os.environ.get("BASE_URL", "https://api.cerebras.ai/v1")

# Get API keys from environment variables
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise ValueError("CEREBRAS_API_KEY environment variable is not set")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)

logger.info("Configuration:")
logger.info(f"WORKING_DIR: {WORKING_DIR}")
logger.info(f"LLM_MODEL: {LLM_MODEL}")
logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
logger.info(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# LLM model function
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    try:
        return await openai_complete_if_cache(
            model=LLM_MODEL,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=CEREBRAS_BASE_URL,
            api_key=CEREBRAS_API_KEY,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Error in LLM model function: {str(e)}")
        raise

# Gemini embedding function
async def embedding_func(texts: List[str]) -> np.ndarray:
    try:
        client = genai.Client()
        embeddings_list = []
        
        for text in texts:
            if not text.strip():  # Skip empty texts
                continue
                
            try:
                result = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=text
                )
                # Ensure we're getting valid embeddings
                if hasattr(result, 'embeddings') and result.embeddings is not None:
                    embeddings_list.append(result.embeddings)
                else:
                    logger.warning(f"No embeddings returned for text: {text[:100]}...")
                    continue
            except Exception as e:
                logger.error(f"Error embedding text: {str(e)}")
                continue
        
        if not embeddings_list:
            raise ValueError("No valid embeddings were generated")
            
        return np.array(embeddings_list)
    except Exception as e:
        logger.error(f"Error in embedding function: {str(e)}")
        raise

async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    try:
        embedding = await embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        logger.info(f"Embedding dimension: {embedding_dim}")
        return embedding_dim
    except Exception as e:
        logger.error(f"Error getting embedding dimension: {str(e)}")
        raise

async def main():
    try:
        # Initialize RAG instance
        embedding_dim = await get_embedding_dim()
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
                func=embedding_func,
            ),
        )

        # Example: Load content from a file
        try:
            with open("./input.txt", "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    logger.error("Input file is empty")
                    return
                logger.info("Inserting content into RAG...")
                rag.insert(content)
        except FileNotFoundError:
            logger.warning("input.txt not found. Please create a file with your content.")
            return
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            return

        # Test different search modes
        query = "What are the main concepts in this content?"
        search_modes = ["naive", "local", "global", "hybrid"]
        
        logger.info("\nTesting different search modes:")
        for mode in search_modes:
            try:
                logger.info(f"\n{mode.upper()} Search Results:")
                result = rag.query(query, param=QueryParam(mode=mode))
                logger.info(result)
            except Exception as e:
                logger.error(f"Error in {mode} search: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 