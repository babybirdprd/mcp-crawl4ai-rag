"""
MCP server for web crawling with Crawl4AI and RAG using Gemini Embeddings.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
Content is stored in Supabase with Gemini embeddings for RAG.
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
from utils import get_supabase_client, add_documents_to_supabase, search_documents, initialize_gemini

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Initialize Gemini client globally (or handle API key errors early)
try:
    initialize_gemini()
except ValueError as e:
    print(f"Error initializing Gemini: {e}")
    print("Please ensure GEMINI_API_KEY is set in your .env file.")
    # Depending on desired behavior, you might exit here or let tools fail later
    # exit(1)

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
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()

    # Initialize Supabase client
    supabase_client = get_supabase_client()

    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client
        )
    finally:
        # Clean up the crawler
        await crawler.__aexit__(None, None, None)

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG (using Gemini Embeddings) and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", 8051)) # Ensure port is integer
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
    try:
        resp = requests.get(sitemap_url, timeout=10) # Added timeout
        resp.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        urls = []

        try:
            # Use 'lxml' if available for better parsing, fallback to 'xml'
            try:
                import lxml.etree as ET
                parser = ET.XMLParser(recover=True) # Tolerate some errors
                tree = ET.fromstring(resp.content, parser=parser)
            except ImportError:
                tree = ElementTree.fromstring(resp.content)

            # Handle namespaces more robustly
            namespace_pattern = re.compile(r'({.*})?(urlset|sitemapindex)')
            root_match = namespace_pattern.match(tree.tag)
            if not root_match:
                print(f"Warning: Unexpected root element '{tree.tag}' in sitemap {sitemap_url}")
                return []

            ns = root_match.group(1) or '' # Extract namespace if present

            # Check if it's a sitemap index
            if tree.tag.endswith('sitemapindex'):
                sitemap_urls = [loc.text for loc in tree.findall(f'.//{ns}loc')]
                nested_urls = []
                # Recursively parse sitemaps listed in the index
                # Consider adding limits or async processing for large indexes
                for sub_sitemap_url in sitemap_urls:
                    nested_urls.extend(parse_sitemap(sub_sitemap_url))
                return nested_urls
            elif tree.tag.endswith('urlset'):
                urls = [loc.text for loc in tree.findall(f'.//{ns}loc')]
                return urls
            else:
                 print(f"Warning: Unknown root element type in sitemap {sitemap_url}")
                 return []

        except ElementTree.ParseError as e:
            print(f"Error parsing sitemap XML from {sitemap_url}: {e}")
            # Try regex as a fallback for malformed XML
            urls = re.findall(r'<loc>(.*?)</loc>', resp.text)
            if urls:
                 print(f"Warning: Used regex fallback for parsing sitemap {sitemap_url}")
                 return urls
            else:
                 return []
        except Exception as e:
            print(f"Unexpected error parsing sitemap {sitemap_url}: {e}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error fetching sitemap {sitemap_url}: {e}")
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
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(chunk_text)

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
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Supabase using Gemini embeddings.

    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is chunked, embedded using Google's Gemini model, and stored in Supabase
    for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl

    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler: AsyncWebCrawler = ctx.request_context.lifespan_context.crawler
        supabase_client: Client = ctx.request_context.lifespan_context.supabase_client

        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)

        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)

        if result.success and result.markdown:
            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)

            if not chunks:
                 return json.dumps({
                    "success": True,
                    "url": url,
                    "message": "Page crawled successfully, but no content chunks generated after processing.",
                    "content_length": len(result.markdown),
                    "links_count": {
                        "internal": len(result.links.get("internal", [])),
                        "external": len(result.links.get("external", []))
                    }
                }, indent=2)

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
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__) if asyncio.current_task() else "unknown"
                metadatas.append(meta)

            # Add to Supabase (uses Gemini embeddings internally)
            add_documents_to_supabase(supabase_client, urls, chunk_numbers, contents, metadatas)

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
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message or "Crawling failed or produced no markdown content."
            }, indent=2)
    except Exception as e:
        print(f"Error in crawl_single_page for {url}: {e}") # Log server-side
        return json.dumps({
            "success": False,
            "url": url,
            "error": f"An unexpected error occurred: {type(e).__name__}"
        }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase using Gemini embeddings.

    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs found within (including nested sitemaps) in parallel.
    - For text files (.txt): Directly retrieves and processes the content.
    - For regular webpages: Recursively crawls internal links up to the specified depth.

    All crawled content is chunked, embedded using Google's Gemini model, and stored in Supabase
    for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 5000)

    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler and Supabase client from the context
        crawler: AsyncWebCrawler = ctx.request_context.lifespan_context.crawler
        supabase_client: Client = ctx.request_context.lifespan_context.supabase_client

        crawl_results = []
        crawl_type = "unknown"
        processed_urls = set() # Keep track of URLs processed to avoid duplicates from sitemap/recursive overlap

        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            crawl_type = "text_file"
            print(f"Detected text file: {url}")
            crawl_results = await crawl_markdown_file(crawler, url)
            if crawl_results:
                processed_urls.add(crawl_results[0]['url']) # Add the single URL processed

        elif is_sitemap(url):
            crawl_type = "sitemap"
            print(f"Detected sitemap: {url}")
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "crawl_type": crawl_type,
                    "error": "No URLs found in sitemap or sitemap could not be parsed."
                }, indent=2)
            print(f"Found {len(sitemap_urls)} URLs in sitemap {url}. Crawling batch...")
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            processed_urls.update(doc['url'] for doc in crawl_results)

        else:
            crawl_type = "webpage"
            print(f"Detected webpage (recursive crawl): {url}")
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            processed_urls.update(doc['url'] for doc in crawl_results)

        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "crawl_type": crawl_type,
                "error": "Crawling completed, but no processable content was found."
            }, indent=2)

        # Process results and store in Supabase
        urls_to_store = []
        chunk_numbers_to_store = []
        contents_to_store = []
        metadatas_to_store = []
        chunk_count = 0
        pages_processed_count = 0

        print(f"Processing {len(crawl_results)} crawled pages...")
        for doc in crawl_results:
            source_url = doc.get('url')
            md = doc.get('markdown')

            if not source_url or not md:
                print(f"Skipping invalid document: {doc.get('url', 'N/A')}")
                continue

            pages_processed_count += 1
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)

            if not chunks:
                print(f"No chunks generated for URL: {source_url}")
                continue # Skip if no chunks were generated

            for i, chunk in enumerate(chunks):
                urls_to_store.append(source_url)
                chunk_numbers_to_store.append(i)
                contents_to_store.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = urlparse(source_url).netloc
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__) if asyncio.current_task() else "unknown"
                metadatas_to_store.append(meta)

                chunk_count += 1

        if not contents_to_store:
             return json.dumps({
                "success": True, # Crawl might have succeeded but yielded no storable content
                "url": url,
                "crawl_type": crawl_type,
                "pages_crawled": pages_processed_count,
                "chunks_stored": 0,
                "message": "Crawling finished, but no content chunks were generated or stored.",
                "urls_processed": sorted(list(processed_urls))[:10] + (["..."] if len(processed_urls) > 10 else [])
            }, indent=2)


        print(f"Storing {chunk_count} chunks from {pages_processed_count} pages in Supabase...")
        # Add to Supabase (uses Gemini embeddings internally)
        # IMPORTANT: Adjust this batch size for more speed if you want! Just don't overwhelm your system or the embedding API ;)
        batch_size = 50 # Increased batch size, monitor Gemini API limits
        add_documents_to_supabase(supabase_client, urls_to_store, chunk_numbers_to_store, contents_to_store, metadatas_to_store, batch_size=batch_size)
        print("Storage complete.")

        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_processed": pages_processed_count,
            "chunks_stored": chunk_count,
            "urls_processed": sorted(list(processed_urls))[:10] + (["..."] if len(processed_urls) > 10 else [])
        }, indent=2)
    except Exception as e:
        print(f"Error in smart_crawl_url for {url}: {e}") # Log server-side
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        return json.dumps({
            "success": False,
            "url": url,
            "error": f"An unexpected error occurred: {type(e).__name__}"
        }, indent=2)


async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.

    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file

    Returns:
        List containing a single dictionary with URL and markdown content, or empty list on failure.
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False) # Bypass cache for files

    try:
        result = await crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            print(f"Successfully crawled text file: {url}")
            return [{'url': url, 'markdown': result.markdown}]
        else:
            print(f"Failed to crawl {url}: {result.error_message or 'No markdown content'}")
            return []
    except Exception as e:
        print(f"Exception during crawl_markdown_file for {url}: {e}")
        return []


async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content for successful crawls.
    """
    if not urls:
        return []

    unique_urls = sorted(list(set(urls))) # Deduplicate and sort for consistency
    print(f"Starting batch crawl for {len(unique_urls)} unique URLs...")

    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    # Consider adjusting dispatcher settings based on expected load and system resources
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=80.0, # Slightly higher threshold
        check_interval=0.5, # Check more frequently
        max_session_permit=max_concurrent
    )

    results_list = []
    try:
        results = await crawler.arun_many(urls=unique_urls, config=crawl_config, dispatcher=dispatcher)
        success_count = 0
        for r in results:
            if r.success and r.markdown:
                results_list.append({'url': r.url, 'markdown': r.markdown})
                success_count += 1
            # else:
            #     print(f"Batch crawl failed for {r.url}: {r.error_message or 'No markdown'}")
        print(f"Batch crawl finished. Success: {success_count}/{len(unique_urls)}")
        return results_list
    except Exception as e:
        print(f"Exception during crawl_batch: {e}")
        return []


async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth (0 means only crawl start_urls)
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content for all successfully crawled pages.
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=80.0,
        check_interval=0.5,
        max_session_permit=max_concurrent
    )

    visited = set()
    results_all = []
    start_domains = set(urlparse(u).netloc for u in start_urls) # Crawl only within start domains

    def normalize_url(url):
        # Normalize: remove fragment, ensure scheme, lowercase domain
        parsed = urlparse(url)
        scheme = parsed.scheme or 'http' # Assume http if missing
        netloc = parsed.netloc.lower()
        path = parsed.path
        query = parsed.query
        # Rebuild without fragment
        normalized = f"{scheme}://{netloc}{path}"
        if query:
            normalized += f"?{query}"
        return normalized

    # Normalize start URLs and filter out external ones initially
    current_urls = set()
    for u in start_urls:
        norm_u = normalize_url(u)
        if urlparse(norm_u).netloc in start_domains:
             current_urls.add(norm_u)
        else:
            print(f"Skipping initial URL {u} as it's outside start domains: {start_domains}")


    print(f"Starting recursive crawl. Depth: {max_depth}, Start URLs: {len(current_urls)}")

    for depth in range(max_depth + 1): # +1 because depth 0 is the start URLs
        urls_to_crawl = sorted([url for url in current_urls if url not in visited])
        if not urls_to_crawl:
            print(f"No new URLs to crawl at depth {depth}. Stopping.")
            break

        print(f"Depth {depth}: Crawling {len(urls_to_crawl)} URLs...")
        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()
        success_count = 0

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url) # Mark as visited even if failed

            if result.success and result.markdown:
                success_count += 1
                results_all.append({'url': result.url, 'markdown': result.markdown}) # Store original URL

                # Only look for links if we haven't reached max depth
                if depth < max_depth:
                    for link in result.links.get("internal", []):
                        try:
                            next_url_raw = link.get("href")
                            if next_url_raw:
                                next_url = normalize_url(next_url_raw)
                                # Check if internal (same domain) and not visited
                                if urlparse(next_url).netloc in start_domains and next_url not in visited:
                                    next_level_urls.add(next_url)
                        except Exception as link_e:
                             print(f"Error processing link '{link.get('href', 'N/A')}' from {norm_url}: {link_e}")


        print(f"Depth {depth}: Finished. Success: {success_count}/{len(urls_to_crawl)}. Found {len(next_level_urls)} new internal URLs.")
        current_urls = next_level_urls

    print(f"Recursive crawl finished. Total pages crawled successfully: {len(results_all)}")
    return results_all


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources (domains) based on unique source metadata values stored in Supabase.

    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with the list of available sources
    """
    try:
        # Get the Supabase client from the context
        supabase_client: Client = ctx.request_context.lifespan_context.supabase_client

        # Efficiently get distinct sources using a view or function if performance is critical.
        # For moderate amounts of data, selecting the metadata column is acceptable.
        # Consider adding pagination or limits if the table grows very large.
        result = supabase_client.from_('crawled_pages')\
            .select('metadata->>source', count='exact')\
            .neq('metadata->>source', 'null')\
            .execute()

        # Use a set to efficiently track unique sources
        unique_sources = set()

        # Extract the source values from the result using a set for uniqueness
        if result.data:
            for item in result.data:
                # The query directly selects the source string
                source = item.get('source')
                if source:
                    unique_sources.add(source)

        # Convert set to sorted list for consistent output
        sources = sorted(list(unique_sources))

        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        print(f"Error in get_available_sources: {e}") # Log server-side
        return json.dumps({
            "success": False,
            "error": f"An unexpected error occurred: {type(e).__name__}"
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: Optional[str] = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content using Gemini embeddings.

    This tool searches the vector database (Supabase) for content relevant to the query using
    vector similarity with embeddings generated by Google's Gemini model.
    Results can be optionally filtered by source domain.

    Use the `get_available_sources` tool first if the user wants to query specific documentation.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com'). If None or empty, no source filter is applied.
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client: Client = ctx.request_context.lifespan_context.supabase_client

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source.strip()} # Ensure clean source string

        # Perform the search (uses Gemini embeddings internally for the query)
        results = search_documents(
            client=supabase_client,
            query=query,
            match_count=match_count,
            filter_metadata=filter_metadata
        )

        # Format the results
        formatted_results = []
        if results: # Check if results is not None or empty
            for result in results:
                # Safely access keys, providing defaults if missing
                formatted_results.append({
                    "url": result.get("url", "N/A"),
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "similarity": result.get("similarity", 0.0)
                })

        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source if source and source.strip() else None, # Report actual filter used
            "results": formatted_results,
            "count": len(formatted_results)
        }, indent=2)
    except Exception as e:
        print(f"Error in perform_rag_query for query '{query}': {e}") # Log server-side
        return json.dumps({
            "success": False,
            "query": query,
            "error": f"An unexpected error occurred: {type(e).__name__}"
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
        exit(1)

if __name__ == "__main__":
    # Basic check for essential env vars
    required_vars = ["GEMINI_API_KEY", "SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please ensure they are set in your .env file or environment.")
        exit(1)

    asyncio.run(main())
