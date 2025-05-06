"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
It uses Voyage AI for embeddings and includes rate limiting for the free tier.
"""
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
# Make sure to import the async versions from utils
from utils import get_supabase_client, add_documents_to_supabase, search_documents, DEFAULT_EMBEDDING_BATCH_SIZE

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    supabase_client: Client

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.

    Args:
        server: The FastMCP server instance

    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )

    # Initialize the crawler
    print("Initializing AsyncWebCrawler...")
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    print("Crawler initialized.")

    # Initialize Supabase client
    print("Initializing Supabase client...")
    supabase_client = get_supabase_client()
    print("Supabase client initialized.")

    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client
        )
    finally:
        # Clean up the crawler
        print("Cleaning up crawler...")
        await crawler.__aexit__(None, None, None)
        print("Crawler cleaned up.")

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI and Voyage AI (Rate Limited)",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8051")) # Ensure port is int
)

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.

    Args:
        url: URL to check

    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.

    Args:
        url: URL to check

    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.

    Args:
        sitemap_url: URL of the sitemap

    Returns:
        List of URLs found in the sitemap
    """
    print(f"Parsing sitemap: {sitemap_url}")
    try:
        resp = requests.get(sitemap_url, timeout=30) # Add timeout
        urls = []

        if resp.status_code == 200:
            try:
                # Added namespace handling for common sitemap formats
                namespaces = {
                    'sitemap': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                    'xhtml': 'http://www.w3.org/1999/xhtml' # Handle potential xhtml links if needed
                }
                tree = ElementTree.fromstring(resp.content)
                # Find 'loc' elements within the sitemap namespace
                urls = [loc.text for loc in tree.findall('.//sitemap:loc', namespaces)]
                print(f"Found {len(urls)} URLs in sitemap.")
            except ElementTree.ParseError as e:
                print(f"Error parsing sitemap XML: {e}. Trying without namespace.")
                # Fallback for sitemaps without explicit namespace
                try:
                    tree = ElementTree.fromstring(resp.content)
                    urls = [loc.text for loc in tree.findall('.//{*}loc')]
                    print(f"Found {len(urls)} URLs in sitemap (fallback).")
                except Exception as fallback_e:
                     print(f"Error parsing sitemap XML (fallback): {fallback_e}")
            except Exception as e:
                 print(f"Error parsing sitemap XML: {e}")
        else:
            print(f"Failed to fetch sitemap: Status code {resp.status_code}")

        return urls
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap URL {sitemap_url}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error parsing sitemap {sitemap_url}: {e}")
        return []


def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        # Ensure the code block marker isn't right at the beginning
        if code_block > chunk_size * 0.1: # Check if marker is reasonably far in
             # Find the *end* of the code block if the marker is an opening one
             if chunk[code_block:].count('```') % 2 != 0:
                 next_code_block_end = text.find('```', start + code_block + 3)
                 if next_code_block_end != -1 and next_code_block_end < end:
                     # If the closing ``` is within the original chunk, break after it
                     end = next_code_block_end + 3
                 else:
                     # If closing ``` is outside, break before the opening ```
                     end = start + code_block
             else:
                 # If it's a closing marker or balanced, break before it
                 end = start + code_block

        # If no suitable code block break, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break + 2 # Include the newlines

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1 # Break after the period

        # Extract chunk and clean it up
        final_chunk = text[start:end].strip()
        if final_chunk:
            chunks.append(final_chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.

    Args:
        chunk: Markdown chunk

    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    # Limit number of headers stored?
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers[:5]]) # Store first 5 headers
    if len(headers) > 5:
        header_str += "; ..."

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase.

    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is chunked, embedded (respecting rate limits), and stored in Supabase.

    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl

    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    print(f"Received crawl_single_page request for URL: {url}")
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

        # Crawl the page
        print(f"  Starting crawl...")
        result = await crawler.arun(url=url, config=run_config)
        print(f"  Crawl finished. Success: {result.success}")

        if result.success and result.markdown:
            # Chunk the content
            print(f"  Chunking content (length: {len(result.markdown)})...")
            chunks = smart_chunk_markdown(result.markdown)
            print(f"  Content chunked into {len(chunks)} pieces.")

            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = urlparse(url).netloc
                # Get the name of the current async task/function if possible
                try:
                     task_name = asyncio.current_task().get_name()
                except Exception:
                     task_name = "crawl_single_page" # Fallback
                meta["crawl_time"] = task_name
                metadatas.append(meta)

            # Add to Supabase (this now handles batching and rate limiting)
            print(f"  Adding {len(contents)} chunks to Supabase...")
            await add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas)
            print(f"  Finished adding documents to Supabase.")

            return json.dumps({
                "success": True,
                "url": url,
                "chunks_stored": len(chunks),
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            error_msg = result.error_message if hasattr(result, 'error_message') else "Unknown crawl error"
            print(f"  Crawl failed or no markdown content found. Error: {error_msg}")
            return json.dumps({
                "success": False,
                "url": url,
                "error": error_msg
            }, indent=2)
    except Exception as e:
        print(f"  Unhandled exception in crawl_single_page: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return json.dumps({
            "success": False,
            "url": url,
            "error": f"Unhandled exception: {str(e)}"
        }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.

    Detects URL type (sitemap, text file, webpage) and applies appropriate crawling.
    Content is chunked, embedded (respecting rate limits), and stored in Supabase.

    Args:
        ctx: The MCP server provided context
        url: URL to crawl (webpage, sitemap.xml, or .txt file)
        max_depth: Max recursion depth for webpages (default: 3)
        max_concurrent: Max concurrent browser sessions (default: 10)
        chunk_size: Max size of content chunks (default: 5000 chars)

    Returns:
        JSON string with crawl summary and storage information
    """
    print(f"Received smart_crawl_url request for URL: {url}")
    try:
        # Get the crawler and Supabase client from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        crawl_results = []
        crawl_type = "unknown"

        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            crawl_type = "text_file"
            print(f"  Detected type: {crawl_type}. Crawling file...")
            crawl_results = await crawl_markdown_file(crawler, url)
        elif is_sitemap(url):
            crawl_type = "sitemap"
            print(f"  Detected type: {crawl_type}. Parsing sitemap...")
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                print("  No URLs found in sitemap.")
                return json.dumps({
                    "success": False, "url": url, "error": "No URLs found in sitemap"
                }, indent=2)
            print(f"  Found {len(sitemap_urls)} URLs. Starting batch crawl (max_concurrent={max_concurrent})...")
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
        else:
            crawl_type = "webpage"
            print(f"  Detected type: {crawl_type}. Starting recursive crawl (max_depth={max_depth}, max_concurrent={max_concurrent})...")
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)

        print(f"  Crawl finished. Found {len(crawl_results)} pages with content.")

        if not crawl_results:
            return json.dumps({
                "success": False, "url": url, "error": "No content found after crawling"
            }, indent=2)

        # Process results and store in Supabase
        urls_all = []
        chunk_numbers_all = []
        contents_all = []
        metadatas_all = []
        total_chunks = 0

        print(f"  Processing {len(crawl_results)} crawled pages for chunking and metadata...")
        for doc in crawl_results:
            source_url = doc.get('url', 'unknown_url')
            md = doc.get('markdown', '')
            if not md:
                print(f"    Skipping page {source_url} due to empty markdown.")
                continue

            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)
            # print(f"    Chunked {source_url} into {len(chunks)} pieces.")

            for i, chunk in enumerate(chunks):
                urls_all.append(source_url)
                chunk_numbers_all.append(i)
                contents_all.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = urlparse(source_url).netloc
                meta["crawl_type"] = crawl_type
                try:
                     task_name = asyncio.current_task().get_name()
                except Exception:
                     task_name = "smart_crawl_url" # Fallback
                meta["crawl_time"] = task_name
                metadatas_all.append(meta)

                total_chunks += 1

        print(f"  Total chunks created: {total_chunks}. Adding to Supabase...")
        # Add to Supabase (handles batching and rate limiting)
        # Use the default batch size from utils unless overridden
        await add_documents_to_supabase(
            supabase_client, urls_all, chunk_numbers_all, contents_all, metadatas_all
        )
        print(f"  Finished adding documents to Supabase.")

        crawled_urls_summary = [doc['url'] for doc in crawl_results[:5]]
        if len(crawl_results) > 5:
            crawled_urls_summary.append("...")

        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(crawl_results),
            "chunks_stored": total_chunks,
            "urls_crawled_sample": crawled_urls_summary
        }, indent=2)
    except Exception as e:
        print(f"  Unhandled exception in smart_crawl_url: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "success": False,
            "url": url,
            "error": f"Unhandled exception: {str(e)}"
        }, indent=2)


async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.
    """
    print(f"    Crawling text file: {url}")
    crawl_config = CrawlerRunConfig() # Default config is fine

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        print(f"    Successfully crawled text file: {url}")
        return [{'url': url, 'markdown': result.markdown}]
    else:
        error_msg = result.error_message if hasattr(result, 'error_message') else "Unknown crawl error"
        print(f"    Failed to crawl {url}: {error_msg}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.
    """
    print(f"    Starting batch crawl for {len(urls)} URLs (max_concurrent={max_concurrent})...")
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    # Use MemoryAdaptiveDispatcher for better resource management
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    successful_results = [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]
    failed_count = len(urls) - len(successful_results)
    print(f"    Batch crawl finished. Success: {len(successful_results)}, Failed/No Content: {failed_count}")
    return successful_results

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.
    """
    print(f"    Starting recursive crawl from {start_urls} (max_depth={max_depth}, max_concurrent={max_concurrent})...")
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()
    results_all = []
    # Use a queue for breadth-first search
    queue = asyncio.Queue()
    for url in start_urls:
        await queue.put((normalize_url(url), 0)) # Add (url, depth)

    while not queue.empty():
        current_url, depth = await queue.get()

        if current_url in visited or depth >= max_depth:
            queue.task_done()
            continue

        if not current_url.startswith(('http://', 'https://')):
             print(f"    Skipping invalid URL scheme: {current_url}")
             visited.add(current_url) # Mark as visited to avoid re-queueing
             queue.task_done()
             continue

        print(f"    Crawling (Depth {depth}): {current_url}")
        visited.add(current_url)

        # Crawl the single URL
        # Note: arun_many might be more efficient if we collect URLs per depth level,
        # but this approach is simpler for strict depth limiting.
        results = await crawler.arun_many(urls=[current_url], config=run_config, dispatcher=dispatcher) # Use arun_many even for one URL

        if results:
            result = results[0] # Get the first (only) result
            if result.success and result.markdown:
                print(f"      Success. Content length: {len(result.markdown)}. Found {len(result.links.get('internal', []))} internal links.")
                results_all.append({'url': result.url, 'markdown': result.markdown})
                # Add internal links to the queue for the next level
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link.get("href"))
                    if next_url and next_url not in visited:
                        # Basic check to stay on the same domain (optional, adjust if needed)
                        if urlparse(next_url).netloc == urlparse(start_urls[0]).netloc:
                             await queue.put((next_url, depth + 1))
                        # else:
                        #     print(f"      Skipping off-domain link: {next_url}")
            else:
                 error_msg = result.error_message if hasattr(result, 'error_message') else "Unknown crawl error"
                 print(f"      Failed or no content. Error: {error_msg}")
        else:
            print(f"      No result returned for {current_url}")


        queue.task_done()
        # Optional: Add a small delay between individual page crawls if needed
        # await asyncio.sleep(0.1)

    print(f"    Recursive crawl finished. Total pages with content: {len(results_all)}")
    return results_all

def normalize_url(url):
    """Normalize URL by removing fragment."""
    if not url: return None
    try:
        # Basic cleaning: remove fragment and trailing slash
        clean_url = urldefrag(url.strip())[0]
        if clean_url.endswith('/'):
            clean_url = clean_url[:-1]
        return clean_url
    except Exception:
        # Handle potential errors if url is not a valid string or format
        print(f"      Error normalizing URL: {url}")
        return None


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources (unique domains) from the stored metadata.

    Useful for discovering what content is available for RAG filtering.

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with the list of available sources
    """
    print("Received get_available_sources request.")
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Efficiently get distinct sources using Supabase features if possible,
        # otherwise fallback to fetching metadata.
        # This might require a specific function or view in Supabase for best performance.
        # Fallback: Fetch metadata and process in Python.
        print("  Querying Supabase for distinct sources...")
        # Limit the query to avoid fetching too much data if the table is huge
        result = supabase_client.from_('crawled_pages')\
            .select('metadata->source', count='exact')\
            .limit(10000) # Adjust limit as needed
            .execute()

        unique_sources = set()
        count = 0

        if hasattr(result, 'data') and result.data:
            for item in result.data:
                # The result structure might vary based on Supabase version/client
                # Adapt this based on actual output of `select('metadata->source')`
                source = item.get('source') # Direct access if 'metadata->source' works
                if not source and 'metadata' in item: # Fallback if nested
                    source = item.get('metadata', {}).get('source')

                if source:
                    unique_sources.add(source)
            # Get count from response if available and accurate
            count = getattr(result, 'count', len(unique_sources))
            # If count seems off (e.g., due to limit), use set length
            if count is None or (result.data and len(result.data) == 10000):
                 count = len(unique_sources)

        else:
             print(f"  No data received from Supabase or error in response: {getattr(result, 'error', 'N/A')}")


        sources = sorted(list(unique_sources))
        print(f"  Found {len(sources)} unique sources (reported count: {count}).")

        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources) # Return actual unique count found
        }, indent=2)
    except Exception as e:
        print(f"  Error in get_available_sources: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: Optional[str] = None, match_count: int = 5) -> str:
    """
    Perform RAG query on stored content using vector similarity search.

    Searches the vector database for content relevant to the query.
    Optionally filters results by source domain.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter (e.g., 'example.com')
        match_count: Max number of results (default: 5)

    Returns:
        JSON string with the search results
    """
    print(f"Received perform_rag_query request. Query: '{query[:50]}...', Source: {source}, Count: {match_count}")
    try:
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source.strip()}
            print(f"  Applying source filter: {filter_metadata}")

        # Perform the search (now async)
        results = await search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )

        # Format the results
        formatted_results = []
        if results: # Check if results is not None or empty
            for result in results:
                formatted_results.append({
                    "url": result.get("url"),
                    "content": result.get("content"), # Consider truncating content for brevity?
                    "metadata": result.get("metadata"),
                    "similarity": result.get("similarity")
                })
        print(f"  Returning {len(formatted_results)} results.")

        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        print(f"  Error in perform_rag_query: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

async def main():
    transport = os.getenv("TRANSPORT", "sse").lower()
    print(f"Starting MCP server with {transport} transport...")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    elif transport == 'stdio':
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()
    else:
        print(f"Error: Invalid TRANSPORT '{transport}'. Use 'sse' or 'stdio'.")
        print("Defaulting to SSE.")
        await mcp.run_sse_async()


if __name__ == "__main__":
    # Ensure event loop policy is set for Windows if needed
    # if os.name == 'nt':
    #    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"\nServer encountered critical error: {e}")
        import traceback
        traceback.print_exc()