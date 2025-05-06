"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import asyncio # Added for sleep
from typing import List, Dict, Any, Optional
import json
from supabase import create_client, Client
from urllib.parse import urlparse
import voyageai

# Initialize Voyage AI client
# It will automatically pick up the VOYAGE_API_KEY environment variable
try:
    vo = voyageai.Client()
    VOYAGE_EMBEDDING_MODEL = "voyage-code-3"
    VOYAGE_EMBEDDING_DIMENSION = 1024 # Default for voyage-code-3
    # Rate Limits (Free Tier)
    VOYAGE_RPM_LIMIT = 3
    VOYAGE_TPM_LIMIT = 10000
    # Calculate delay needed between requests (in seconds)
    REQUEST_DELAY = 60 / VOYAGE_RPM_LIMIT + 1 # Add 1s buffer
except Exception as e:
    print(f"Error initializing Voyage AI client: {e}. Ensure VOYAGE_API_KEY is set.")
    vo = None
    VOYAGE_EMBEDDING_MODEL = None
    VOYAGE_EMBEDDING_DIMENSION = 1024 # Keep default dimension for fallback
    REQUEST_DELAY = 21 # Fallback delay

# Define a smaller default batch size to respect token limits
DEFAULT_EMBEDDING_BATCH_SIZE = 5

def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.

    Returns:
        Supabase client instance
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables")

    return create_client(url, key)

# Note: create_embeddings_batch itself remains synchronous as vo.embed is sync
# The rate limiting (delay) is handled by the caller (add_documents_to_supabase)
def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a single API call using Voyage AI.
    WARNING: This function does NOT handle rate limiting itself. The caller must implement delays.

    Args:
        texts: List of texts to create embeddings for (should respect batch size limits)

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if not texts:
        print("No texts provided to create_embeddings_batch.")
        return []
    if not vo or not VOYAGE_EMBEDDING_MODEL:
        print("Voyage AI client not initialized. Returning zero vectors.")
        return [[0.0] * VOYAGE_EMBEDDING_DIMENSION for _ in range(len(texts))]

    # Estimate token count (very rough estimate: ~4 chars/token)
    # total_chars = sum(len(t) for t in texts)
    # estimated_tokens = total_chars / 4
    # print(f"Estimating {estimated_tokens} tokens for {len(texts)} texts.")
    # if estimated_tokens > VOYAGE_TPM_LIMIT:
    #     print(f"Warning: Estimated tokens ({estimated_tokens}) might exceed TPM limit ({VOYAGE_TPM_LIMIT}) for a single request if called too frequently.")

    try:
        # Using input_type=None as we are embedding chunks directly
        result = vo.embed(
            texts=texts,
            model=VOYAGE_EMBEDDING_MODEL,
            input_type=None, # Or "document" if specifically embedding docs for retrieval
            truncation=True # Truncate long texts
        )
        # print(f"Voyage AI API call successful. Tokens used: {result.total_tokens}") # Voyage client doesn't return total_tokens directly in v0.6.0 result object
        return result.embeddings
    except Exception as e:
        print(f"Error creating batch embeddings with Voyage AI: {e}")
        # Return zero vectors if there's an error
        return [[0.0] * VOYAGE_EMBEDDING_DIMENSION for _ in range(len(texts))]

# Needs to be async because it calls the async create_embeddings_batch
async def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using Voyage AI's API.
    Handles rate limiting implicitly via create_embeddings_batch caller if needed,
    but a single call is unlikely to trigger limits.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding
    """
    # A single embedding call is less likely to hit rate limits,
    # but uses the batch function for consistency.
    # No explicit sleep here, assuming single calls are infrequent.
    try:
        # Pass as a list to the batch function
        embeddings = create_embeddings_batch([text])
        return embeddings[0] if embeddings else [0.0] * VOYAGE_EMBEDDING_DIMENSION
    except Exception as e:
        print(f"Error creating single embedding with Voyage AI: {e}")
        # Return zero vector if there's an error
        return [0.0] * VOYAGE_EMBEDDING_DIMENSION

# Make async to allow for asyncio.sleep
async def add_documents_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE # Use smaller default
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches, respecting rate limits.
    Deletes existing records with the same URLs before inserting to prevent duplicates.

    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        batch_size: Size of each batch for embedding and insertion (should be small, e.g., 5)
    """
    if batch_size > DEFAULT_EMBEDDING_BATCH_SIZE:
        print(f"Warning: Provided batch_size ({batch_size}) is larger than recommended ({DEFAULT_EMBEDDING_BATCH_SIZE}) for Voyage AI free tier. Adjusting to {DEFAULT_EMBEDDING_BATCH_SIZE}.")
        batch_size = DEFAULT_EMBEDDING_BATCH_SIZE

    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))

    # Delete existing records for these URLs first (single operation)
    if unique_urls:
        try:
            print(f"Deleting existing records for {len(unique_urls)} URLs...")
            client.table("crawled_pages").delete().in_("url", unique_urls).execute()
            print("Deletion complete.")
        except Exception as e:
            print(f"Error during bulk delete: {e}. Records might not be deleted.")
            # Decide if you want to proceed or raise an error
            # For now, we'll print the error and continue insertion

    total_docs = len(contents)
    print(f"Starting insertion of {total_docs} chunks in batches of {batch_size}...")

    # Process in batches to avoid memory issues and API rate limits
    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        current_batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size

        print(f"Processing batch {current_batch_num}/{total_batches} (indices {i} to {batch_end-1})...")

        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]

        # --- Rate Limiting Point ---
        # Create embeddings for the entire batch at once (API Call)
        print(f"  Creating embeddings for {len(batch_contents)} chunks...")
        batch_embeddings = create_embeddings_batch(batch_contents)
        print(f"  Embeddings created.")

        if len(batch_embeddings) != len(batch_contents):
             print(f"  Error: Mismatch between number of contents ({len(batch_contents)}) and embeddings ({len(batch_embeddings)}). Skipping batch.")
             # Add delay even if skipping insert, as API call was made
             print(f"  Waiting for {REQUEST_DELAY:.1f} seconds due to rate limit...")
             await asyncio.sleep(REQUEST_DELAY)
             continue # Skip to the next batch

        batch_data = []
        for j in range(len(batch_contents)):
            # Extract metadata fields
            chunk_size = len(batch_contents[j])

            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": batch_contents[j],
                "metadata": {
                    "chunk_size": chunk_size,
                    **batch_metadatas[j]
                },
                "embedding": batch_embeddings[j] # Use the generated embedding
            }
            batch_data.append(data)

        # Insert batch into Supabase
        try:
            print(f"  Inserting {len(batch_data)} records into Supabase...")
            client.table("crawled_pages").insert(batch_data).execute()
            print(f"  Batch {current_batch_num}/{total_batches} inserted successfully.")
        except Exception as e:
            print(f"  Error inserting batch {current_batch_num} into Supabase: {e}")
            # Decide how to handle insert errors (e.g., log, retry, skip)

        # --- Rate Limiting Point ---
        # Wait *after* processing the batch (embedding + insert) before starting the next
        # Only sleep if there are more batches to process
        if batch_end < total_docs:
            print(f"  Waiting for {REQUEST_DELAY:.1f} seconds due to rate limit...")
            await asyncio.sleep(REQUEST_DELAY)

    print("All batches processed.")


# Make async because it calls async create_embedding
async def search_documents(
    client: Client,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity.

    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter

    Returns:
        List of matching documents
    """
    # Create embedding for the query
    # A single query embed is unlikely to hit limits, uses the async version
    print(f"Creating embedding for query: '{query[:50]}...'")
    query_embedding = await create_embedding(query)
    print("Query embedding created.")

    # Execute the search using the match_crawled_pages function
    try:
        print(f"Searching Supabase (match_count={match_count}, filter={filter_metadata})...")
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count
        }

        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params['filter'] = filter_metadata  # Pass the dictionary directly

        result = client.rpc('match_crawled_pages', params).execute()
        print(f"Search complete. Found {len(result.data)} results.")
        return result.data
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []
