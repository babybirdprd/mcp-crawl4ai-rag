"""
Utility functions for the Crawl4AI MCP server.
Includes retries and better logging for embedding generation.
"""
import os
import asyncio
import time # Added for retries
import traceback # Added for detailed error logging
from typing import List, Dict, Any, Optional
import json
from supabase import create_client, Client
from urllib.parse import urlparse
import voyageai

# Initialize Voyage AI client
try:
    vo = voyageai.Client()
    VOYAGE_EMBEDDING_MODEL = "voyage-code-3"
    VOYAGE_EMBEDDING_DIMENSION = 1024 # Default for voyage-code-3
    # Rate Limits (Free Tier)
    VOYAGE_RPM_LIMIT = 3
    VOYAGE_TPM_LIMIT = 10000
    # Calculate delay needed between requests (in seconds) - Increase buffer slightly
    REQUEST_DELAY = (60 / VOYAGE_RPM_LIMIT) + 5 # Add 5s buffer -> ~28 seconds
    print(f"Voyage AI Request Delay set to: {REQUEST_DELAY:.1f} seconds")
except Exception as e:
    print(f"Error initializing Voyage AI client: {e}. Ensure VOYAGE_API_KEY is set.")
    vo = None
    VOYAGE_EMBEDDING_MODEL = None
    VOYAGE_EMBEDDING_DIMENSION = 1024
    REQUEST_DELAY = 25 # Fallback delay

# Define a smaller default batch size to respect token limits
DEFAULT_EMBEDDING_BATCH_SIZE = 5
# Retry configuration
MAX_EMBEDDING_RETRIES = 5
INITIAL_RETRY_DELAY = 2 # seconds

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
# The inter-batch rate limiting (delay) is handled by the caller (add_documents_to_supabase)
# This function now includes intra-batch retries for API errors.
def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts using Voyage AI, with retries.
    WARNING: Inter-batch rate limiting must be handled by the caller.

    Args:
        texts: List of texts to create embeddings for (should respect batch size limits)

    Returns:
        List of embeddings (each embedding is a list of floats).
        Returns zero vectors for the batch if all retries fail.
    """
    if not texts:
        print("No texts provided to create_embeddings_batch.")
        return []
    if not vo or not VOYAGE_EMBEDDING_MODEL:
        print("Voyage AI client not initialized. Returning zero vectors.")
        return [[0.0] * VOYAGE_EMBEDDING_DIMENSION for _ in range(len(texts))]

    retries = 0
    current_delay = INITIAL_RETRY_DELAY
    while retries < MAX_EMBEDDING_RETRIES:
        try:
            # Using input_type=None as we are embedding chunks directly
            result = vo.embed(
                texts=texts,
                model=VOYAGE_EMBEDDING_MODEL,
                input_type=None,
                truncation=True
            )
            # print(f"Voyage AI API call successful on attempt {retries + 1}.")
            return result.embeddings # Success!

        except Exception as e:
            retries += 1
            print(f"--- Error creating batch embeddings (Attempt {retries}/{MAX_EMBEDDING_RETRIES}) ---")
            print(f"    Error Type: {type(e).__name__}")
            print(f"    Error Args: {e.args}")
            # Log full traceback for unexpected errors on the last attempt
            if retries == MAX_EMBEDDING_RETRIES:
                 print("    Max retries reached. Returning zero vectors for this batch.")
                 print("    Last error traceback:")
                 traceback.print_exc() # Log the full traceback
            else:
                # Check if it's a known rate limit error type if library provides one
                # (voyageai library might not expose specific error types well)
                # For now, retry on any exception, but wait longer
                print(f"    Waiting for {current_delay} seconds before retrying...")
                time.sleep(current_delay)
                current_delay *= 3 # Exponential backoff

    # If all retries fail, return zero vectors
    return [[0.0] * VOYAGE_EMBEDDING_DIMENSION for _ in range(len(texts))]


# Needs to be async because it calls the async create_embeddings_batch caller if needed,
# but a single call is unlikely to trigger limits.
async def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using Voyage AI's API.
    Handles rate limiting implicitly via create_embeddings_batch caller if needed,
    but a single call is unlikely to trigger limits. Includes retries.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding, or a zero vector on failure.
    """
    # A single embedding call is less likely to hit rate limits,
    # but uses the batch function for consistency and retries.
    # No explicit sleep here, assuming single calls are infrequent.
    try:
        # Pass as a list to the batch function
        embeddings = create_embeddings_batch([text]) # This now has retries
        return embeddings[0] if embeddings else [0.0] * VOYAGE_EMBEDDING_DIMENSION
    except Exception as e:
        # This outer catch is less likely to be hit now due to retries inside batch func
        print(f"Error creating single embedding even after retries: {e}")
        traceback.print_exc()
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
    Uses retries for embedding generation.

    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        batch_size: Size of each batch for embedding and insertion (should be small, e.g., 5)
    """
    if batch_size > DEFAULT_EMBEDDING_BATCH_SIZE * 2: # Allow slightly larger if specified, but warn
        print(f"Warning: Provided batch_size ({batch_size}) is significantly larger than recommended ({DEFAULT_EMBEDDING_BATCH_SIZE}) for Voyage AI free tier. Rate limiting errors are more likely.")
        # Keep user's batch_size but they are warned. Consider capping it:
        # batch_size = DEFAULT_EMBEDDING_BATCH_SIZE * 2

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

    total_docs = len(contents)
    print(f"Starting insertion of {total_docs} chunks in batches of {batch_size}...")
    successful_embeddings = 0
    failed_embeddings = 0

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
        # Create embeddings for the entire batch at once (API Call with Retries)
        print(f"  Creating embeddings for {len(batch_contents)} chunks...")
        batch_embeddings = create_embeddings_batch(batch_contents) # This function now handles retries
        print(f"  Embeddings generation attempt finished for batch.")

        # Check for failed embeddings (zero vectors)
        batch_failed_count = sum(1 for emb in batch_embeddings if all(v == 0.0 for v in emb))
        failed_embeddings += batch_failed_count
        successful_embeddings += len(batch_contents) - batch_failed_count
        if batch_failed_count > 0:
             print(f"  WARNING: Failed to generate embeddings for {batch_failed_count}/{len(batch_contents)} chunks in this batch (received zero vectors). Check previous logs for errors.")

        if len(batch_embeddings) != len(batch_contents):
             # This case should be less likely now, but keep as a safeguard
             print(f"  Error: Mismatch between number of contents ({len(batch_contents)}) and embeddings ({len(batch_embeddings)}). Skipping DB insert for batch.")
             # Add delay even if skipping insert, as API call was attempted
             print(f"  Waiting for {REQUEST_DELAY:.1f} seconds due to rate limit...")
             await asyncio.sleep(REQUEST_DELAY)
             continue # Skip to the next batch

        batch_data = []
        for j in range(len(batch_contents)):
            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": batch_contents[j], # Consider adding flag if embedding failed?
                "metadata": {
                    "chunk_size": len(batch_contents[j]),
                    **batch_metadatas[j]
                    # Optional: Add flag indicating embedding success/failure
                    # "embedding_success": not all(v == 0.0 for v in batch_embeddings[j])
                },
                "embedding": batch_embeddings[j] # Use the generated embedding (might be zeros)
            }
            batch_data.append(data)

        # Insert batch into Supabase
        if batch_data:
            try:
                print(f"  Inserting {len(batch_data)} records into Supabase...")
                client.table("crawled_pages").insert(batch_data).execute()
                print(f"  Batch {current_batch_num}/{total_batches} inserted successfully.")
            except Exception as e:
                print(f"  Error inserting batch {current_batch_num} into Supabase: {e}")
                print(f"  Failed data sample: {batch_data[0] if batch_data else 'N/A'}") # Log sample data on failure
                traceback.print_exc() # Log full traceback for insert errors
        else:
             print("  Skipping insertion as batch_data is empty.")


        # --- Rate Limiting Point ---
        # Wait *after* processing the batch (embedding + insert) before starting the next
        # Only sleep if there are more batches to process
        if batch_end < total_docs:
            print(f"--- Waiting for {REQUEST_DELAY:.1f} seconds before next batch (Rate Limit Delay) ---")
            await asyncio.sleep(REQUEST_DELAY)

    print("="*40)
    print("All batches processed.")
    print(f"Embedding Summary: Success={successful_embeddings}, Failed={failed_embeddings}")
    print("="*40)


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
    # Create embedding for the query (includes retries)
    print(f"Creating embedding for query: '{query[:50]}...'")
    query_embedding = await create_embedding(query)
    if all(v == 0.0 for v in query_embedding):
        print("  WARNING: Failed to generate embedding for the query. Search results will be meaningless.")
        # Optionally return empty list immediately if query embedding fails
        # return []
    else:
        print("  Query embedding created.")


    # Execute the search using the match_crawled_pages function
    try:
        print(f"Searching Supabase (match_count={match_count}, filter={filter_metadata})...")
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count
        }
        if filter_metadata:
            params['filter'] = filter_metadata

        result = client.rpc('match_crawled_pages', params).execute()
        print(f"Search complete. Found {len(result.data)} results.")
        return result.data
    except Exception as e:
        print(f"Error searching documents: {e}")
        traceback.print_exc()
        return []