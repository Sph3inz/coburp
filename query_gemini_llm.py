#!/usr/bin/env python3
"""
Example script to insert and query graph data using the Gemini LLM services.

This script:
• Initializes the GeminiEmbeddingService and GeminiLLMService.
• Creates a GraphRAG instance with these Gemini services.
• Inserts sample documents into GraphRAG.
• Performs queries showing both "only_context" and "with_references" modes.
• Formats the answer with inline references.
"""

import json
import argparse
import os
import shutil
from fast_graphrag import GraphRAG, QueryParam
from gemini_llm import GeminiEmbeddingService, GeminiLLMService

# Replace with your actual Gemini API key.
GEMINI_API_KEY = "AIzaSyAwjj3FkVASfvXfY71jla-fAhlpGF9Bdnc"

parser = argparse.ArgumentParser(description="Query script for Gemini LLM GraphRAG")
parser.add_argument(
    "--reset-index",
    action="store_true",
    help="Clear GraphRAG index to ensure embedding consistency."
)
args = parser.parse_args()

if args.reset_index:
    if os.path.exists("./graphdata"):
         print("[Info] Resetting GraphRAG index by clearing the ./graphdata directory.")
         shutil.rmtree("./graphdata")
    else:
         print("[Info] No existing index found to reset.")

# Initialize the Gemini services
embedding_service = GeminiEmbeddingService(
    api_key=GEMINI_API_KEY,
    model="all-mpnet-base-v2"  # Using a sentence-transformers model
)
llm_service = GeminiLLMService(
    model="gemini-2.0-flash-lite-preview-02-05",  # Gemini model name
    api_key=GEMINI_API_KEY,
    max_retries=3,
    retry_delay=2.0,
    rate_limit_max_retries=5,
    temperature=0.7
)

# Create a GraphRAG instance that uses the Gemini services
grag = GraphRAG(
    working_dir="./graphdata",         # Directory to store data
    domain="Example Domain",           # Domain description
    example_queries="This is a sample query example.",
    entity_types=["Article", "Document"],
    config=GraphRAG.Config(
        llm_service=llm_service,
        embedding_service=embedding_service
    )
)

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def query_loop():
    """
    This loop continuously prompts for a query and returns the LLM's answer.
    """
    while True:
        try:
            user_query = input("\nEnter your query (or type 'exit' to quit): ")
            if user_query.strip().lower() in ["exit", "quit"]:
                print("Exiting query loop.")
                break

            answer = grag.query(user_query)
            print("\n[Response]:")
            print(answer.response)
        except Exception as e:
            print(f"Error processing query: {e}")

def main():
    # Start interactive query loop.
    query_loop()

if __name__ == "__main__":
    main() 