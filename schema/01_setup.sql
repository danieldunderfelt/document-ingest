CREATE EXTENSION IF NOT EXISTS vector;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document_id UUID REFERENCES documents(id),
    project_id TEXT,
    file_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

DROP TABLE IF EXISTS document_chunks;

-- Create document_chunks table with vector support
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id),
    project_id TEXT,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    requires_embedding BOOLEAN DEFAULT true,
    embedding vector(768),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create HNSW index for vector similarity search
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx ON document_chunks 
USING hnsw (embedding vector_cosine_ops)
WITH (
    m = 16,        -- max number of connections per layer
    ef_construction = 64  -- size of dynamic candidate list for construction
);
