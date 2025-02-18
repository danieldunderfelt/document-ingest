import os
from typing import Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from hatchet_sdk import ClientConfig, Hatchet, Context
import uvicorn
import asyncio
import logging
from uuid import uuid4
import httpx
import asyncpg
import json
import base64
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    AcceleratorOptions,
    AcceleratorDevice
)
from docling.document_converter import DocumentConverter, PdfFormatOption
import mimetypes
from chonkie import SDPMChunker, SemanticChunker, SentenceTransformerEmbeddings
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global variables
hatchet_client = None
db_pool = None

# Initialize hatchet client at module level
token = os.getenv("HATCHET_CLIENT_TOKEN", "")

hatchet_client = Hatchet(
    config=ClientConfig(
        server_url=os.getenv("HATCHET_SERVER_URL", "http://hatchet-lite:8888"),
        token=token
    ),
    debug=True
)

async def init_db_schema():
    """Initialize database schema if not exists"""
    schema_name = os.getenv("INGEST_DB_SCHEMA", "public")
    
    # Read schema file
    with open("schema/01_setup.sql", "r") as f:
        schema_sql = f.read()
    
    # Replace schema if needed
    if schema_name != "public":
        schema_sql = f"CREATE SCHEMA IF NOT EXISTS {schema_name};\nSET search_path TO {schema_name};\n" + schema_sql
    
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(schema_sql)
            logger.info(f"Schema initialized successfully in schema: {schema_name}")
    except Exception as e:
        logger.error(f"Error initializing schema: {e}")
        raise

async def create_db_pool():
    """Create database connection pool"""
    schema_name = os.getenv("INGEST_DB_SCHEMA", "public")
    
    # Create pool without schema parameter and disable statement cache
    pool = await asyncpg.create_pool(
        host=os.getenv("INGEST_DB_HOST"),
        port=os.getenv("INGEST_DB_PORT"),
        database=os.getenv("INGEST_DB_NAME"),
        user=os.getenv("INGEST_DB_USER"),
        password=os.getenv("INGEST_DB_PASSWORD"),
        statement_cache_size=0  # Disable statement cache for pgbouncer compatibility
    )
    
    # Set schema for all connections in the pool
    async with pool.acquire() as conn:
        await conn.execute(f'SET search_path TO {schema_name}')
    
    return pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app startup and shutdown"""

    global db_pool
    
    logger.info("Starting up ingest service...")
    
    # Initialize database pool
    db_pool = await create_db_pool()
    
    # Initialize schema
    await init_db_schema()
    
    # Start Hatchet worker (using existing client)
    worker = hatchet_client.worker(
        "ingest-worker",
        max_runs=4  # Limit concurrent step runs
    )
    worker.register_workflow(DocumentIngestWorkflow())
    # worker.register_workflow(EmbeddingGenerationWorkflow())
    worker_task = asyncio.create_task(worker.async_start())
    
    yield
    
    # Shutdown
    logger.info("Shutting down ingest service...")
    worker_task.cancel()
    await db_pool.close()
    
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

class IngestRequest(BaseModel):
    file_id: str
    project_id: Optional[str] = None
    preprocess: Optional[bool] = False
    preprocess_prompt: Optional[str] = None

@app.post("/ingest/file")
async def ingest_file(
    file: UploadFile = File(...),
    project_id: Optional[str] = None,
    preprocess: Optional[bool] = False,
    preprocess_prompt: Optional[str] = None
):
    """Endpoint to receive files for ingestion"""
    try:
        file_id = str(uuid4())
        file_content = await file.read()
        
        # Store file temporarily
        temp_path = f"/tmp/{file_id}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(file_content)

        # Push event to trigger workflow (synchronously)
        hatchet_client.event.push(
            "document:ingest",  # Event key
            {  # Event payload
                "file_id": file_id,
                "file_path": temp_path,
                "file_name": file.filename,
                "project_id": project_id,
                "preprocess": preprocess,
                "preprocess_prompt": preprocess_prompt
            }
        )

        return {"status": "accepted", "file_id": file_id}

    except Exception as e:
        logger.error(f"Error in file ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hatchet_client.workflow(on_events=["document:ingest"], timeout="10m")
class DocumentIngestWorkflow:
    async def handle_document_versioning(self, conn, file_id: str, file_name: str, project_id: str, 
                                       input_format: InputFormat, mime_type: str) -> str:
        """Handle document versioning and return the document ID to use.
        
        Args:
            conn: Database connection
            file_id: Generated UUID for new document
            file_name: Name of the uploaded file
            project_id: Project ID
            input_format: Document format
            mime_type: MIME type of the document
            
        Returns:
            str: Document ID to use (either existing or new)
        """
        existing_doc = await conn.fetchrow("""
            SELECT id, metadata 
            FROM documents 
            WHERE file_name = $1 AND project_id = $2
        """, file_name, project_id)

        if existing_doc:
            # Use existing document id
            doc_id = existing_doc["id"]
            
            # Update existing document
            await conn.execute("""
                UPDATE documents 
                SET metadata = $1, 
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = $2
            """, 
            json.dumps({
                "format": input_format.value,
                "mime_type": mime_type,
                "version": (existing_doc["metadata"].get("version", 0) + 1 
                          if existing_doc["metadata"] else 1)
            }),
            doc_id)
            
            # Delete old chunks
            await conn.execute("""
                DELETE FROM document_chunks 
                WHERE document_id = $1
            """, doc_id)
            
            logger.info(f"Updating existing document: {doc_id}")
            
            return doc_id
        else:
            # Create new document record
            await conn.execute("""
                INSERT INTO documents (id, project_id, file_name, metadata)
                VALUES ($1, $2, $3, $4)
            """, 
            file_id, 
            project_id, 
            file_name,
            json.dumps({
                "format": input_format.value,
                "mime_type": mime_type,
                "version": 1
            })
            )

            logger.info(f"Created new document: {file_id}")
            return file_id

    @hatchet_client.step(timeout="10m")
    async def parse_and_chunk(self, context: Context):
        """Parse document and create chunks"""
        from docling.document_converter import DocumentConverter
        from tokenizers import Tokenizer

        inputs = context.workflow_input()
        file_path = inputs["file_path"]
        file_id = inputs["file_id"]
        
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4, 
                device=AcceleratorDevice.AUTO
            )

            # Configure document converter
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )

            # Detect file type
            mime_type, _ = mimetypes.guess_type(inputs["file_name"])
            
            # Map MIME types to Docling InputFormat
            mime_to_format = {
                'application/pdf': InputFormat.PDF,
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': InputFormat.DOCX,
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': InputFormat.XLSX,
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': InputFormat.PPTX,
                'text/plain': InputFormat.ASCIIDOC,
                'text/html': InputFormat.HTML,
                'application/xhtml+xml': InputFormat.HTML,
                'text/csv': InputFormat.CSV,
                'image/png': InputFormat.IMAGE,
                'image/jpeg': InputFormat.IMAGE,
                'image/tiff': InputFormat.IMAGE,
                'image/bmp': InputFormat.IMAGE,
                'application/xml': InputFormat.XML_JATS,
                'application/json': InputFormat.JSON_DOCLING,
                None: InputFormat.ASCIIDOC
            }

            input_format = mime_to_format.get(mime_type)
            if not input_format:
                raise ValueError(f"Unsupported file type: {mime_type}")

            # Parse document with Docling
            doc_result = doc_converter.convert(file_path)
            text_content = doc_result.document.export_to_markdown()

       
            # Clean up the content
            def clean_content(text: str) -> str:
                # Replace multiple dots with a single one
                text = re.sub(r'\.{2,}', '.', text)
                # Replace multiple spaces and newlines with single ones
                text = re.sub(r'\s+', ' ', text)
                # Remove empty lines
                text = re.sub(r'^\s*$\n', '', text, flags=re.MULTILINE)
                # Clean up table of contents style lines
                text = re.sub(r'\|[^|]*\.\.\.[^|]*\|', '', text)
                return text.strip()

            # Clean the content before chunking
            text_content = clean_content(text_content)

            # Initialize embeddings model and chunker
            embeddings = SentenceTransformerEmbeddings("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
            
            tokenizer = Tokenizer.from_pretrained("gpt2")
            chunker = SDPMChunker(
                tokenizer=tokenizer,
                embedding_model=embeddings,
                chunk_size=512,          # Maximum tokens per chunk
                overlap=32,              # Overlap for context preservation
                min_chunk_size=128,      # Increased minimum chunk size
                min_sentences=2,         # Require at least 2 sentences per chunk
                min_characters_per_sentence=32,  # Minimum characters per sentence
                similarity_window=2,           # Higher threshold for better semantic coherence
                mode="window",       # Use cumulative mode for better context
                skip_window=5,
                trust_remote_code=True,
                prompt_name="passage"
            )

            # Create chunks
            chunks = chunker(text_content)
            
            # Filter out empty or whitespace-only chunks
            chunks = [chunk for chunk in chunks 
                     if chunk.text.strip() 
                     and len(chunk.text.strip()) >= 50]

            # Handle document versioning
            async with db_pool.acquire() as conn:
                file_id = await self.handle_document_versioning(
                    conn=conn,
                    file_id=file_id,
                    file_name=inputs["file_name"],
                    project_id=inputs["project_id"],
                    input_format=input_format,
                    mime_type=mime_type
                )

            # Prepare chunks for database insertion
            chunk_records = []
            for i, chunk in enumerate(chunks):
                # Average the sentence embeddings to get chunk embedding
                sentence_embeddings = [s.embedding for s in chunk.sentences if s.embedding is not None]
                if sentence_embeddings:
                    chunk_embedding = np.mean(sentence_embeddings, axis=0)
                else:
                    # If no sentence embeddings, generate one for the whole chunk
                    chunk_embedding = embeddings.embed([chunk.text])[0]
                
                chunk_records.append({
                    "content": chunk.text,
                    "token_count": chunk.token_count,
                    "document_id": file_id,
                    "project_id": inputs["project_id"],
                    "chunk_index": i,
                    "requires_embedding": False,
                    "embedding": json.dumps(chunk_embedding.tolist()),  # Convert to JSON string
                    "metadata": json.dumps({
                        "format": input_format.value,
                        "chunk_type": "markdown"
                    })
                })

            # Insert chunks into database
            async with db_pool.acquire() as conn:
                await conn.executemany("""
                    INSERT INTO document_chunks (
                        content, token_count, document_id, project_id, 
                        chunk_index, requires_embedding, embedding, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, [(
                    r["content"], 
                    r["token_count"], 
                    r["document_id"],
                    r["project_id"],
                    r["chunk_index"],
                    r["requires_embedding"],
                    r["embedding"],
                    r["metadata"]
                ) for r in chunk_records])

            # Clean up temporary file
            os.remove(file_path)

            return {
                "status": "success",
                "document_id": file_id,
                "chunks_created": len(chunk_records),
                "format": input_format.value
            }

        except Exception as e:
            logger.error(f"Error in parse_and_chunk: {e}")
            raise

class QueryEmbeddingRequest(BaseModel):
    text: str
    prompt_name: Optional[str] = "query"  # Default to "query" for search queries

@app.post("/embed/query")
async def embed_query(request: QueryEmbeddingRequest):
    """Generate embeddings for a search query using the same model as documents"""
    try:
        # Initialize embeddings model
        embeddings = SentenceTransformerEmbeddings(
            "nomic-ai/nomic-embed-text-v2-moe",
            trust_remote_code=True  # Add this to allow loading remote model code
        )
        
        # Generate embedding for the query
        embedding = embeddings.embed([request.text])[0]
        
        return {
            "embedding": embedding.tolist(),  # Convert numpy array to list for JSON serialization
            "model": "nomic-embed-text-v2-moe",
            "prompt_name": request.prompt_name
        }
        
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
