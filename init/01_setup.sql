-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgai;

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    project_id TEXT,
    file_name TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create document_chunks table with vector support
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id TEXT REFERENCES documents(id),
    project_id TEXT,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    requires_embedding BOOLEAN DEFAULT true,
    embedding vector(1024),  -- voyage-3-large uses 1024 dimensions
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

-- Set up pgai vectorizer for the chunks table
SELECT ai.create_vectorizer(
    'document_chunks'::regclass,
    destination => 'document_chunks',
    embedding => ai.embedding_voyageai(
        'voyage-3-large',  -- upgraded to large model
        1024              -- larger dimension size
    ),
    source_column => 'content',
    destination_column => 'embedding',
    trigger_condition => 'requires_embedding = true'
);

-- Create function to mark chunks as requiring re-embedding
CREATE OR REPLACE FUNCTION mark_for_reembedding()
RETURNS TRIGGER AS $$
BEGIN
    NEW.requires_embedding := true;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically mark updated content for re-embedding
CREATE TRIGGER content_update_trigger
    BEFORE UPDATE OF content ON document_chunks
    FOR EACH ROW
    EXECUTE FUNCTION mark_for_reembedding(); 