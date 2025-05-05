<h1 align="center">Crawl4AI RAG MCP Server (with Gemini Embeddings)</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants using Google Gemini</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com), [Supabase](https://supabase.com/), and [Google Gemini Embeddings](https://ai.google.dev/docs/embeddings) for providing AI agents and AI coding assistants with advanced web crawling and Retrieval-Augmented Generation (RAG) capabilities.

With this MCP server, you can <b>scrape anything</b> and then <b>use that knowledge anywhere</b> for RAG, leveraging Google's state-of-the-art embedding models.

The primary goal is to bring this MCP server into [Archon](https://github.com/coleam00/Archon) as I evolve it to be more of a knowledge engine for AI coding assistants to build AI agents. This version uses the Gemini embedding model and will be improved upon greatly soon, especially making it more configurable so you can use different embedding models (including local ones via Ollama).

## Overview

This MCP server provides tools that enable AI agents to crawl websites, store content in a vector database (Supabase) using Gemini embeddings, and perform RAG over the crawled content. It follows the best practices for building MCP servers based on the [Mem0 MCP server template](https://github.com/coleam00/mcp-mem0/) I provided on my channel previously.

## Vision

The Crawl4AI RAG MCP server is just the beginning. Here's where we're headed:

1.  **Integration with Archon**: Building this system directly into [Archon](https://github.com/coleam00/Archon) to create a comprehensive knowledge engine for AI coding assistants to build better AI agents.
2.  **Multiple Embedding Models**: Expanding beyond Gemini to support a variety of embedding models, including the ability to run everything locally with Ollama for complete control and privacy.
3.  **Advanced RAG Strategies**: Implementing sophisticated retrieval techniques like contextual retrieval, late chunking, and others to move beyond basic "naive lookups" and significantly enhance the power and precision of the RAG system, especially as it integrates with Archon.
4.  **Enhanced Chunking Strategy**: Implementing a Context 7-inspired chunking approach that focuses on examples and creates distinct, semantically meaningful sections for each chunk, improving retrieval precision.
5.  **Performance Optimization**: Increasing crawling and indexing speed to make it more realistic to "quickly" index new documentation to then leverage it within the same prompt in an AI coding assistant.

## Features

-   **Smart URL Detection**: Automatically detects and handles different URL types (regular webpages, sitemaps, text files)
-   **Recursive Crawling**: Follows internal links within the same domain to discover content
-   **Parallel Processing**: Efficiently crawls multiple pages simultaneously
-   **Content Chunking**: Intelligently splits content by headers and size for better processing
-   **Gemini Embeddings**: Utilizes Google's `embedding-001` model (configurable to experimental models) for state-of-the-art semantic understanding.
-   **Vector Search**: Performs RAG over crawled content using Supabase pgvector, optionally filtering by data source for precision.
-   **Source Retrieval**: Retrieve sources available for filtering to guide the RAG process.
-   **Robust Error Handling**: Includes retries for Gemini API calls and better handling of crawl/parse errors.

## Tools

The server provides four essential web crawling and search tools:

1.  **`crawl_single_page`**: Quickly crawl a single web page, generate Gemini embeddings, and store its content in the vector database.
2.  **`smart_crawl_url`**: Intelligently crawl a full website based on the type of URL provided (sitemap, llms-full.txt, or a regular webpage that needs to be crawled recursively). Embeds content using Gemini.
3.  **`get_available_sources`**: Get a list of all available sources (domains) in the database.
4.  **`perform_rag_query`**: Search for relevant content using semantic search (powered by Gemini embeddings) with optional source filtering.

## Prerequisites

-   [Docker/Docker Desktop](https://www.docker.com/products/docker-desktop/) if running the MCP server as a container (recommended)
-   [Python 3.12+](https://www.python.org/downloads/) if running the MCP server directly through uv
-   [Supabase](https://supabase.com/) project (database for RAG)
-   [Google Gemini API key](https://aistudio.google.com/app/apikey) (for generating embeddings)

## Installation

### Using Docker (Recommended)

1.  Clone this repository:
    ```bash
    git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
    cd mcp-crawl4ai-rag
    ```

2.  Build the Docker image:
    ```bash
    docker build -t mcp/crawl4ai-rag --build-arg PORT=8051 .
    ```

3.  Create a `.env` file based on the configuration section below.

### Using uv directly (no Docker)

1.  Clone this repository:
    ```bash
    git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
    cd mcp-crawl4ai-rag
    ```

2.  Install uv if you don't have it:
    ```bash
    pip install uv
    ```

3.  Create and activate a virtual environment:
    ```bash
    uv venv
    # On Windows:
    .venv\Scripts\activate
    # On Mac/Linux:
    source .venv/bin/activate
    ```

4.  Install dependencies:
    ```bash
    uv pip install -e .
    crawl4ai-setup
    ```

5.  Create a `.env` file based on the configuration section below.

## Database Setup

Before running the server, you need to set up the database with the pgvector extension:

1.  Go to the SQL Editor in your Supabase dashboard (create a new project first if necessary).
2.  Create a new query and paste the contents of `crawled_pages.sql`.
3.  Run the query to create the necessary tables, indexes, and functions. The schema uses `vector(1536)` which is compatible with the configured Gemini embedding dimension.

## Configuration

Create a `.env` file in the project root with the following variables:

```dotenv
# MCP Server Configuration
# TRANSPORT: 'sse' or 'stdio' (defaults to sse)
TRANSPORT=sse
# HOST: Host to bind to for SSE (defaults to 0.0.0.0)
HOST=0.0.0.0
# PORT: Port to listen on for SSE (defaults to 8051)
PORT=8051

# Google Gemini API Configuration
# Get key from https://aistudio.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key

# Supabase Configuration
# Get from your Supabase project settings -> API
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
```

## Running the Server

### Using Docker

Ensure your `.env` file is in the project root.

```bash
docker run --env-file .env -p 8051:8051 mcp/crawl4ai-rag
```

### Using Python (uv)

Ensure your virtual environment is activated and the `.env` file is present.

```bash
uv run src/crawl4ai_mcp.py
```

The server will start and listen on the configured host and port (for SSE) or use stdio. Check the console output for initialization messages and potential errors (like missing API keys).

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport (`TRANSPORT=sse` in `.env`), you can connect to it using this configuration in your MCP client (e.g., Claude Desktop, Windsurf):

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "transport": "sse",
      "url": "http://localhost:8051/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "crawl4ai-rag": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8051/sse"
>     }
>   }
> }
> ```
>
> **Note for Docker users**: If your client is running in a *different* Docker container on the same machine, use `host.docker.internal` instead of `localhost` (e.g., `http://host.docker.internal:8051/sse`). This is common when using this MCP server within n8n running in Docker. If the client is running directly on your host machine, `localhost` is correct.

### Stdio Configuration

If you set `TRANSPORT=stdio` in your `.env` file, configure your MCP client like this:

**Using Python directly:**

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "uv", // Use uv to run
      "args": ["run", "path/to/mcp-crawl4ai-rag/src/crawl4ai_mcp.py"], // Adjust path
      "env": {
        "TRANSPORT": "stdio",
        "GEMINI_API_KEY": "your_gemini_api_key", // Can be set here or rely on .env
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```
*(Ensure the path to `crawl4ai_mcp.py` is correct relative to where the MCP client is running)*

**Using Docker with Stdio:**

```json
{
  "mcpServers": {
    "crawl4ai-rag": {
      "command": "docker",
      "args": ["run", "--rm", "-i", // -i for interactive/stdio
               "--env-file", "path/to/your/.env", // Mount or pass env file
               // OR pass individual env vars:
               // "-e", "TRANSPORT=stdio",
               // "-e", "GEMINI_API_KEY",
               // "-e", "SUPABASE_URL",
               // "-e", "SUPABASE_SERVICE_KEY",
               "mcp/crawl4ai-rag"], // Your built image name
      "env": {
        // Pass keys from client env to docker env if needed,
        // otherwise rely on --env-file
        "TRANSPORT": "stdio",
        "GEMINI_API_KEY": "your_gemini_api_key",
        "SUPABASE_URL": "your_supabase_url",
        "SUPABASE_SERVICE_KEY": "your_supabase_service_key"
      }
    }
  }
}
```
*(Adjust the path to your `.env` file if using `--env-file`)*

## Building Your Own Server

This implementation provides a foundation for building more complex MCP servers with web crawling and Gemini-powered RAG capabilities. To build your own:

1.  Add your own tools by creating methods with the `@mcp.tool()` decorator.
2.  Create your own lifespan function (`@asynccontextmanager`) to manage custom dependencies.
3.  Modify the `utils.py` file for any helper functions you need (e.g., different embedding models, database interactions).
4.  Extend the crawling capabilities by adding more specialized crawlers or post-processing steps.