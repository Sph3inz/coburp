import asyncio
import json
import signal
import sys
from pathlib import Path
import datetime
import aiofiles
from typing import Dict, List, Optional, Set, Tuple, Any
import time
from dataclasses import dataclass, asdict
import logging

from crawl4ai import (
    AsyncWebCrawler, 
    BrowserConfig, 
    CrawlerRunConfig, 
    CacheMode,
    PruningContentFilter,
    DefaultMarkdownGenerator
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('crawler.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class CrawlResult:
    """Represents the result of crawling a single URL."""
    url: str
    title: Optional[str]
    content: str
    crawl_time: str
    success: bool
    error: Optional[str] = None

@dataclass
class CrawlProgress:
    """Tracks crawling progress and manages state."""
    processed_urls: Set[str]
    results: List[CrawlResult]
    start_time: float
    total_urls: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "processed_urls": list(self.processed_urls),
            "results": [asdict(r) for r in self.results],
            "start_time": self.start_time,
            "total_urls": self.total_urls
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CrawlProgress':
        return cls(
            processed_urls=set(data["processed_urls"]),
            results=[CrawlResult(**r) for r in data["results"]],
            start_time=data["start_time"],
            total_urls=data["total_urls"]
        )

class FastCrawler:
    def __init__(self, 
                 batch_size: int = 10,
                 save_interval: int = 5,
                 output_file: str = "crawled_data.json",
                 progress_file: str = "crawler_progress.json",
                 links_per_file: int = 500):
        self.batch_size = batch_size
        self.save_interval = save_interval
        self.output_file = output_file
        self.progress_file = progress_file
        self.progress: Optional[CrawlProgress] = None
        self.last_save_time = 0
        self.links_per_file = links_per_file
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals by saving progress and exiting gracefully."""
        logger.info("\nReceived interrupt signal. Saving progress and shutting down...")
        if self.progress:
            self._save_progress()
            self._save_results()
        sys.exit(0)

    async def _save_progress(self):
        """Save current progress to file."""
        if not self.progress:
            return
        
        async with aiofiles.open(self.progress_file, 'w') as f:
            await f.write(json.dumps(self.progress.to_dict()))
        self.last_save_time = time.time()
        logger.info(f"Progress saved to {self.progress_file}")

    async def _save_results(self):
        """Save crawled results to output files."""
        if not self.progress:
            return
        
        # Calculate how many files we need
        total_results = len(self.progress.results)
        num_files = (total_results + self.links_per_file - 1) // self.links_per_file
        
        for file_num in range(num_files):
            start_idx = file_num * self.links_per_file
            end_idx = min((file_num + 1) * self.links_per_file, total_results)
            current_batch = self.progress.results[start_idx:end_idx]
            
            # Format results as text
            text_content = f"Crawled Content (Part {file_num + 1})\n"
            text_content += f"Last updated: {datetime.datetime.now().isoformat()}\n"
            text_content += f"Documents {start_idx + 1} to {end_idx} of {total_results}\n\n"
            
            # Add each crawled page as a section
            for result in current_batch:
                if result.success and result.content:
                    text_content += "=" * 80 + "\n\n"
                    text_content += f"Title: {result.title or 'Untitled'}\n"
                    text_content += f"URL: {result.url}\n"
                    text_content += f"Crawled: {result.crawl_time}\n\n"
                    text_content += result.content + "\n\n"
            
            # Save as text file with part number
            text_file = self.output_file.replace('.json', f'_part{file_num + 1}.txt')
            async with aiofiles.open(text_file, 'w', encoding='utf-8') as f:
                await f.write(text_content)
            logger.info(f"Text results saved to {text_file}")
        
        # Also save the JSON version for compatibility
        output_data = {
            "documents": [
                {
                    "url": result.url,
                    "title": result.title,
                    "content": result.content,
                    "metadata": {
                        "crawl_time": result.crawl_time,
                        "success": result.success,
                        "error": result.error
                    }
                }
                for result in self.progress.results
            ],
            "metadata": {
                "total_documents": len(self.progress.results),
                "total_urls_processed": len(self.progress.processed_urls),
                "total_files": num_files,
                "links_per_file": self.links_per_file,
                "crawl_start_time": datetime.datetime.fromtimestamp(self.progress.start_time).isoformat(),
                "last_update": datetime.datetime.now().isoformat()
            }
        }
        
        async with aiofiles.open(self.output_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(output_data, indent=2))
        logger.info(f"JSON results saved to {self.output_file}")

    async def _load_progress(self) -> Optional[CrawlProgress]:
        """Load previous progress if it exists."""
        try:
            if Path(self.progress_file).exists():
                async with aiofiles.open(self.progress_file, 'r') as f:
                    data = json.loads(await f.read())
                    return CrawlProgress.from_dict(data)
        except Exception as e:
            logger.warning(f"Could not load previous progress: {e}")
        return None

    async def _crawl_url(self, url: str) -> CrawlResult:
        """Crawl a single URL using crawl4ai."""
        browser_config = BrowserConfig(
            headless=True,
            verbose=False,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport_width=1920,
            viewport_height=1080
        )
        
        # Create content filter for better content extraction
        content_filter = PruningContentFilter(
            threshold=0.5,
            min_word_threshold=50
        )
        
        # Create markdown generator with options
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
            js_code="window.scrollTo(0, document.body.scrollHeight);",  # Scroll to bottom
            wait_until="networkidle",  # Wait for network to be idle
            page_timeout=60000,  # Increase timeout to 60 seconds
            verbose=False,  # Reduce verbose output
            markdown_generator=markdown_generator
        )
        
        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)
                
                if result.success:
                    content = None
                    # Only process successful crawls with actual content
                    if result.markdown_v2 and result.markdown_v2.raw_markdown and result.markdown_v2.raw_markdown.strip():
                        content = result.markdown_v2.raw_markdown
                    elif result.markdown and (isinstance(result.markdown, str) and result.markdown.strip() or 
                                           hasattr(result.markdown, 'raw_markdown') and result.markdown.raw_markdown.strip()):
                        content = result.markdown if isinstance(result.markdown, str) else result.markdown.raw_markdown
                    elif result.fit_markdown and result.fit_markdown.strip():
                        content = result.fit_markdown
                    elif result.cleaned_html and result.cleaned_html.strip():
                        content = f"""
# {result.metadata.get('title', 'Untitled Page')}

{result.cleaned_html}
"""
                    
                    if content and content.strip():
                        return CrawlResult(
                            url=url,
                            title=result.metadata.get("title") if result.metadata else None,
                            content=content,
                            crawl_time=datetime.datetime.now().isoformat(),
                            success=True
                        )
                    else:
                        # No usable content found, treat as failed silently
                        return CrawlResult(
                            url=url,
                            title=None,
                            content="",
                            crawl_time=datetime.datetime.now().isoformat(),
                            success=False,
                            error="No usable content found"
                        )
                
                # Handle failed crawls silently without detailed error messages
                return CrawlResult(
                    url=url,
                    title=None,
                    content="",
                    crawl_time=datetime.datetime.now().isoformat(),
                    success=False,
                    error="Failed to crawl"
                )
                
        except Exception as e:
            # Handle exceptions silently
            return CrawlResult(
                url=url,
                title=None,
                content="",
                crawl_time=datetime.datetime.now().isoformat(),
                success=False,
                error="Failed to crawl"
            )

    async def _process_batch(self, urls: List[str]) -> List[CrawlResult]:
        """Process a batch of URLs in parallel."""
        # Reduce batch size for better reliability
        tasks = [self._crawl_url(url) for url in urls]
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing error: {str(e)}")
        return results

    def _get_new_urls(self, writeups_path: str) -> List[str]:
        """Extract new URLs from writeups.json that haven't been processed."""
        try:
            with open(writeups_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, dict) or "data" not in data:
                return []
                
            new_urls = []
            for entry in data["data"]:
                if isinstance(entry, dict) and "Links" in entry:
                    for link_item in entry["Links"]:
                        if isinstance(link_item, dict) and "Link" in link_item:
                            url = link_item["Link"]
                            if (url and isinstance(url, str) and 
                                url not in self.progress.processed_urls):
                                new_urls.append(url)
                                
            return new_urls
        except Exception as e:
            logger.error(f"Error reading writeups.json: {e}")
            return []

    async def run(self, writeups_path: str):
        """Main crawling loop."""
        # Load previous progress or initialize new
        self.progress = await self._load_progress() or CrawlProgress(
            processed_urls=set(),
            results=[],
            start_time=time.time()
        )
        
        while True:
            try:
                # Get new URLs to process
                new_urls = self._get_new_urls(writeups_path)
                if not new_urls:
                    logger.info("No new URLs to process. Waiting...")
                    await asyncio.sleep(5)
                    continue
                
                self.progress.total_urls = len(new_urls)
                logger.info(f"Found {len(new_urls)} new URLs to process")
                
                # Process URLs in batches
                for i in range(0, len(new_urls), self.batch_size):
                    batch = new_urls[i:i + self.batch_size]
                    results = await self._process_batch(batch)
                    
                    # Update progress
                    self.progress.results.extend(results)
                    self.progress.processed_urls.update(r.url for r in results)
                    
                    # Log progress
                    processed = len(self.progress.processed_urls)
                    total = self.progress.total_urls
                    logger.info(f"Progress: {processed}/{total} URLs processed")
                    
                    # Save progress periodically
                    if time.time() - self.last_save_time >= self.save_interval:
                        await self._save_progress()
                        await self._save_results()
                    
                    # Small delay between batches to prevent overwhelming
                    await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast parallel web crawler with progress saving")
    parser.add_argument("--writeups", type=str, default="writeups.json",
                       help="Path to writeups.json file")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Number of URLs to process in parallel")
    parser.add_argument("--save-interval", type=int, default=5,
                       help="How often to save progress (in seconds)")
    parser.add_argument("--output", type=str, default="crawled_data.json",
                       help="Output file for crawled data")
    parser.add_argument("--progress", type=str, default="crawler_progress.json",
                       help="File to save progress state")
    parser.add_argument("--links-per-file", type=int, default=500,
                       help="Number of links per markdown file")
    
    args = parser.parse_args()
    
    crawler = FastCrawler(
        batch_size=args.batch_size,
        save_interval=args.save_interval,
        output_file=args.output,
        progress_file=args.progress,
        links_per_file=args.links_per_file
    )
    
    try:
        asyncio.run(crawler.run(args.writeups))
    except KeyboardInterrupt:
        logger.info("\nShutting down gracefully...")
        sys.exit(0) 