-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table crawled_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    content text not null,  -- Added content column
    metadata jsonb not null default '{}'::jsonb,  -- Added metadata column
    embedding vector(1024),  -- Changed dimension from 1536 to 1024 for voyage-code-3
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,

    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
-- Note: Consider recreating the index if you change embedding dimensions on existing data
create index on crawled_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_crawled_pages_metadata on crawled_pages using gin (metadata);

CREATE INDEX idx_crawled_pages_source ON crawled_pages ((metadata->>'source'));

-- Create a function to search for documentation chunks
create or replace function match_crawled_pages (
  query_embedding vector(1024), -- Changed dimension from 1536 to 1024
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
    1 - (crawled_pages.embedding <=> query_embedding) as similarity
  from crawled_pages
  where metadata @> filter
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
