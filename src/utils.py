"""
Utility functions for the Crawl4AI MCP server, using Google Gemini for embeddings.
Uses the recommended google-genai library and the experimental embedding model by default.
"""
import os
import time
from typing import List, Dict, Any, Optional
from supabase import create_client, Client
from dotenv import load_dotenv
# Use the new recommended library 'google-genai'
from google import genai

from google.api_core import exceptions as google_exceptions

# --- Constants ---
# Use the experimental SOTA model by default as requested
GEMINI_EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
# Keep dimension consistent with DB schema (experimental model supports 768, 1536, 3072)
# Supabase only supports 1536 dimensions for vector columns
OUTPUT_DIMENSIONALITY = 1536
DEFAULT_EMBEDDING_DIM = OUTPUT_DIMENSIONALITY # Fallback dimension
MAX_GEMINI_BATCH_SIZE = 100 # Gemini API limit for embed_content batch size
RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2

# --- Initialization ---

# Load environment variables (useful if this module is used independently)
load_dotenv()

# Global Gemini client instance (using the new library's client approach)
gemini_client: Optional[genai.Client] = None

def initialize_gemini():
    """Initializes the global Gemini client using google-genai."""
    global gemini_client
    if gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in environment variables")
        # Configure the client instance
        gemini_client = genai.Client(api_key=api_key)
        print(f"Gemini client initialized with model: {GEMINI_EMBEDDING_MODEL}")

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

# --- Embedding Functions ---

def _embed_content_with_retry(
    model: str,
    contents: List[str], # API uses 'contents' for list input
    task_type: str,
    output_dimensionality: Optional[int] = None
) -> EmbedContentResponse:
    """Internal function to call Gemini API using the client with retry logic."""
    global gemini_client
    if gemini_client is None:
        initialize_gemini() # Ensure client is initialized

    # Prepare configuration using types.EmbedContentConfig
    config_kwargs = {'task_type': task_type}
    if output_dimensionality is not None:
        config_kwargs['output_dimensionality'] = output_dimensionality
    config = types.EmbedContentConfig(**config_kwargs)

    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Call embed_content on the client's model attribute
            return gemini_client.models.embed_content(
                model=model,
                contents=contents, # Pass the list of texts
                config=config      # Pass the config object
            )

        except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError) as e:
            print(f"Gemini API rate limit or server error (Attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}. Retrying in {RETRY_DELAY_SECONDS}s...")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print("Max retry attempts reached. Raising error.")
                raise # Re-raise the last exception after final attempt
        except Exception as e:
            print(f"Unexpected error calling Gemini API (Attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            # Depending on the error, you might want to retry or raise immediately
            if attempt < RETRY_ATTEMPTS - 1:
                 time.sleep(RETRY_DELAY_SECONDS)
            else:
                 raise # Re-raise after final attempt for unexpected errors too

def create_embeddings_batch(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = MAX_GEMINI_BATCH_SIZE
) -> List[List[float]]:
    """
    Create embeddings for multiple texts using Gemini, handling batching and retries.

    Args:
        texts: List of texts to create embeddings for.
        task_type: The task type for the embedding model (e.g., "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY").
        batch_size: How many texts to process in each API call.

    Returns:
        List of embeddings (each embedding is a list of floats). Returns empty lists
        of the correct dimension for texts that failed embedding after retries.
    """
    if not texts:
        return []

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        # Pre-fill with error embeddings matching the target dimensionality
        batch_embeddings_list = [[0.0] * OUTPUT_DIMENSIONALITY] * len(batch_texts)

        try:
            # Ensure Gemini client is ready (initialization check is inside _embed_content_with_retry)
            print(f"Requesting Gemini embeddings for batch {i//batch_size + 1} (size {len(batch_texts)}), task: {task_type}, model: {GEMINI_EMBEDDING_MODEL}, dim: {OUTPUT_DIMENSIONALITY}")
            response = _embed_content_with_retry(
                model=GEMINI_EMBEDDING_MODEL,
                contents=batch_texts, # Use 'contents' for the list
                task_type=task_type,
                output_dimensionality=OUTPUT_DIMENSIONALITY
            )

            # Process the response (structure might be slightly different with client)
            # The client response structure usually has an 'embeddings' attribute which is a list of dicts {'embedding': [floats]}
            if response and hasattr(response, 'embeddings') and isinstance(response.embeddings, list):
                 if len(response.embeddings) == len(batch_texts):
                     # Extract the 'embedding' list from each dict
                     batch_embeddings_list = [emb['embedding'] for emb in response.embeddings]
                 else:
                     print(f"Warning: Gemini API returned {len(response.embeddings)} embedding dicts for {len(batch_texts)} texts.")
                     # Fallback logic: Try to match based on order
                     valid_embeddings = [emb['embedding'] for emb in response.embeddings if 'embedding' in emb]
                     for idx in range(min(len(valid_embeddings), len(batch_texts))):
                         batch_embeddings_list[idx] = valid_embeddings[idx]
            else:
                print(f"Warning: Gemini API returned no valid embeddings structure for batch starting at index {i}. Response: {response}")


        except Exception as e:
            print(f"Error creating batch embeddings (batch starting index {i}): {e}. Using default embeddings for this batch.")
            # batch_embeddings_list remains filled with default error embeddings

        all_embeddings.extend(batch_embeddings_list)
        print(f"Processed batch {i//batch_size + 1}. Total embeddings generated: {len(all_embeddings)}")
        # Optional: Add a small delay between batches to help with rate limits
        if len(texts) > batch_size and i + batch_size < len(texts):
             time.sleep(0.5) # 500ms delay

    # Final check: Ensure the number of embeddings matches the number of input texts
    if len(all_embeddings) != len(texts):
        print(f"Error: Mismatch between input texts ({len(texts)}) and generated embeddings ({len(all_embeddings)}). Padding/Truncating.")
        # Pad with default embeddings if necessary
        while len(all_embeddings) < len(texts):
            all_embeddings.append([0.0] * OUTPUT_DIMENSIONALITY)
        # Truncate if too many
        all_embeddings = all_embeddings[:len(texts)]

    return all_embeddings


def create_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """
    Create an embedding for a single text using Gemini.

    Args:
        text: Text to create an embedding for.
        task_type: The task type for the embedding model.

    Returns:
        List of floats representing the embedding, or a default zero vector on error.
    """
    try:
        # Batch size of 1 is handled by create_embeddings_batch
        embeddings = create_embeddings_batch([text], task_type=task_type, batch_size=1)
        return embeddings[0] if embeddings else [0.0] * OUTPUT_DIMENSIONALITY
    except Exception as e:
        print(f"Error creating single embedding: {e}")
        return [0.0] * OUTPUT_DIMENSIONALITY

# --- Supabase Functions ---

def add_documents_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 50 # Match embedding batch size or make independent
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Generates Gemini embeddings for the content before insertion.
    Deletes existing records with the same URLs before inserting to prevent duplicates.

    Args:
        client: Supabase client.
        urls: List of URLs.
        chunk_numbers: List of chunk numbers.
        contents: List of document contents.
        metadatas: List of document metadata.
        batch_size: Size of each batch for embedding and insertion.
    """
    if not contents:
        print("No documents provided to add_documents_to_supabase.")
        return

    # --- 1. Delete existing records for the affected URLs ---
    unique_urls = sorted(list(set(urls)))
    print(f"Deleting existing records for {len(unique_urls)} unique URLs...")
    delete_batch_size = 500 # Supabase might handle larger deletes efficiently
    deleted_count = 0
    try:
        for i in range(0, len(unique_urls), delete_batch_size):
            batch_urls = unique_urls[i:i + delete_batch_size]
            delete_result = client.table("crawled_pages").delete().in_("url", batch_urls).execute()
            # Note: Supabase delete response might not give a precise count easily.
            # We assume success if no exception is raised.
            deleted_count += len(batch_urls) # Approximate count
            print(f"Attempted deletion for batch {i//delete_batch_size + 1} (size {len(batch_urls)}).")
        print(f"Deletion phase complete for ~{deleted_count} URLs.")
    except Exception as e:
        print(f"Error during batch deletion: {e}. Insertion might result in duplicates or conflicts.")
        # Decide whether to proceed or raise the error
        # raise e # Option: Stop execution if deletion fails critically

    # --- 2. Process and Insert new data in batches ---
    print(f"Processing and inserting {len(contents)} chunks in batches of {batch_size}...")
    inserted_count = 0
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))
        print(f"Processing batch {i//batch_size + 1}/{ (len(contents) + batch_size - 1)//batch_size } (indices {i}-{batch_end-1})")

        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]

        # Create embeddings for the batch
        # Use RETRIEVAL_DOCUMENT task type for storing content
        batch_embeddings = create_embeddings_batch(batch_contents, task_type="RETRIEVAL_DOCUMENT", batch_size=batch_size)

        # Check if embedding generation was successful for the batch
        if not batch_embeddings or len(batch_embeddings) != len(batch_contents):
             print(f"Error: Embedding generation failed or returned incorrect count for batch {i}. Skipping insertion for this batch.")
             continue # Skip to the next batch

        batch_data = []
        for j in range(len(batch_contents)):
            # Prepare data for insertion
            # Ensure metadata is serializable JSON
            current_metadata = batch_metadatas[j]
            try:
                # Add chunk size to metadata
                current_metadata["chunk_size"] = len(batch_contents[j])
                # You might want to add other auto-generated metadata here
            except Exception as meta_e:
                print(f"Warning: Could not process metadata for chunk {j} in batch {i}: {meta_e}")
                current_metadata = {"error": "metadata processing failed", **current_metadata}


            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": batch_contents[j],
                "metadata": current_metadata, # Already includes chunk_size
                "embedding": batch_embeddings[j]
            }
            batch_data.append(data)

        # Insert batch into Supabase
        try:
            if batch_data: # Ensure there's data to insert
                insert_result = client.table("crawled_pages").insert(batch_data).execute()
                # Basic check: Supabase insert result often has 'data' list
                if hasattr(insert_result, 'data') and isinstance(insert_result.data, list):
                     inserted_count += len(insert_result.data)
                     print(f"Successfully inserted batch {i//batch_size + 1}. Total inserted so far: {inserted_count}")
                else:
                     # Fallback count if response format is unexpected
                     inserted_count += len(batch_data)
                     print(f"Inserted batch {i//batch_size + 1} (response format unclear). Total inserted approx: {inserted_count}")

            else:
                 print(f"Skipping insertion for batch {i//batch_size + 1} as no data was prepared (likely due to embedding errors).")

        except Exception as e:
            print(f"Error inserting batch {i//batch_size + 1} into Supabase: {e}")
            # Option: Implement per-row insertion fallback or logging failed rows

    print(f"Finished adding documents. Total chunks processed: {len(contents)}, Total chunks inserted: {inserted_count}")


def search_documents(
    client: Client,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity with Gemini embeddings.

    Args:
        client: Supabase client.
        query: Query text.
        match_count: Maximum number of results to return.
        filter_metadata: Optional metadata filter (e.g., {"source": "example.com"}).

    Returns:
        List of matching documents, or empty list on error.
    """
    try:
        # Create embedding for the query using the appropriate task type
        print(f"Creating Gemini embedding for query (task: RETRIEVAL_QUERY, model: {GEMINI_EMBEDDING_MODEL}): '{query[:100]}...'")
        query_embedding = create_embedding(query, task_type="RETRIEVAL_QUERY")

        if not query_embedding or query_embedding == [0.0] * OUTPUT_DIMENSIONALITY:
             print("Error: Failed to generate embedding for the query.")
             return []

        # Prepare parameters for the RPC call
        params = {
            'query_embedding': query_embedding,
            'match_count': match_count,
            'filter': filter_metadata if filter_metadata else {} # Pass empty dict if no filter
        }

        print(f"Executing Supabase RPC 'match_crawled_pages' with match_count={match_count}, filter={params['filter']}")
        result = client.rpc('match_crawled_pages', params).execute()

        if result.data:
            print(f"Found {len(result.data)} potential matches.")
            return result.data
        else:
            # Handle cases where the RPC call itself might have issues (check result status/error if available)
            print("No documents found matching the query criteria.")
            return []

    except Exception as e:
        print(f"Error searching documents for query '{query}': {e}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging RPC errors
        return []

# --- Main guard ---
if __name__ == "__main__":
    # Example usage (optional, for testing)
    print("Utils module loaded. Testing functions...")
    try:
        initialize_gemini()
        print("Gemini client initialized.")

        # Test embedding
        # test_text = "This is a test sentence for Gemini experimental embeddings."
        # embedding = create_embedding(test_text, task_type="SEMANTIC_SIMILARITY")
        # if embedding and embedding != [0.0] * OUTPUT_DIMENSIONALITY:
        #     print(f"Successfully created embedding for test text (first 5 dims): {embedding[:5]}...")
        #     print(f"Embedding dimension: {len(embedding)}")
        # else:
        #     print("Failed to create test embedding.")

        # Test Supabase connection (requires .env)
        # try:
        #     sb_client = get_supabase_client()
        #     print("Supabase client created.")
        #     # Example search (replace with a real query if data exists)
        #     # search_results = search_documents(sb_client, "example query", match_count=2)
        #     # print(f"Example search results: {search_results}")
        # except ValueError as ve:
        #      print(f"Supabase connection test failed: {ve}")
        # except Exception as sb_e:
        #      print(f"Error during Supabase test: {sb_e}")


    except ValueError as ve:
        print(f"Initialization failed: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")