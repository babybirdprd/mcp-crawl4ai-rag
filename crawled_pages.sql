-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(1536),  -- Gemini embeddings can be 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,

    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
-- Using cosine distance as it's common for semantic similarity
create index on crawled_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_crawled_pages_metadata on crawled_pages using gin (metadata);

CREATE INDEX idx_crawled_pages_source ON crawled_pages ((metadata->>'source'));

-- Create a function to search for documentation chunks
create or replace function match_crawled_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    content,
    metadata,
    -- Calculate cosine similarity (1 - cosine distance)
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  from crawled_pages
  where metadata @> filter
  -- Order by cosine distance (ascending, smaller is better)
  order by crawled_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the table
alter table crawled_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on crawled_pages
  for select
  to public
  using (true);