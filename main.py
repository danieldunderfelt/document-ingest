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
    PipelineOptions, 
    EasyOcrOptions,
    TesseractOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
import mimetypes
from chonkie import SDPMChunker, SentenceTransformerEmbeddings
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer

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
    
    # Create pool without schema parameter
    pool = await asyncpg.create_pool(
        host=os.getenv("INGEST_DB_HOST"),
        port=os.getenv("INGEST_DB_PORT"),
        database=os.getenv("INGEST_DB_NAME"),
        user=os.getenv("INGEST_DB_USER"),
        password=os.getenv("INGEST_DB_PASSWORD")
    )
    
    # Set schema for all connections in the pool
    async with pool.acquire() as conn:
        await conn.execute(f'SET search_path TO {schema_name}')
    
    return pool

'''
async def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Voyage API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.getenv('VOYAGE_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "voyage-3-large",
                "input": texts
            }
        )
        
        if response.status_code != 200:
            raise Exception(f"Error generating embeddings: {response.text}")
        
        return [item["embedding"] for item in response.json()["data"]]

@hatchet_client.workflow(on_crons=["*/1 * * * *"])  # Run every minute
class EmbeddingGenerationWorkflow:
    @hatchet_client.step()
    async def generate_pending_embeddings(self, context: Context):
        # Generate embeddings for chunks that need them
        try:
            async with db_pool.acquire() as conn:
                # Fetch chunks that need embeddings
                chunks = await conn.fetch("""
                    SELECT id, content 
                    FROM document_chunks 
                    WHERE requires_embedding = true 
                    LIMIT 100
                """)
                
                if chunks:
                    # Generate embeddings
                    embeddings = await generate_embeddings([chunk["content"] for chunk in chunks])
                    
                    # Update chunks with embeddings
                    for chunk, embedding in zip(chunks, embeddings):
                        await conn.execute("""
                            UPDATE document_chunks 
                            SET embedding = $1, requires_embedding = false 
                            WHERE id = $2
                        """, embedding, chunk["id"])
                    
                    logger.info(f"Generated embeddings for {len(chunks)} chunks")
                    return {"chunks_processed": len(chunks)}
                
                return {"chunks_processed": 0}
                
        except Exception as e:
            logger.error(f"Error in embedding generation task: {e}")
            raise
'''

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

@hatchet_client.workflow(on_events=["document:ingest"])  # Listen for document:ingest events
class DocumentIngestWorkflow:
    @hatchet_client.step()
    async def parse_and_chunk(self, context: Context):
        """Parse document and create chunks"""
        from docling.document_converter import DocumentConverter
        from tokenizers import Tokenizer

        inputs = context.workflow_input()
        file_path = inputs["file_path"]
        file_id = inputs["file_id"]
        
        try:
            # Configure Docling with prefetched models and multiple OCR options
            pipeline_options = PipelineOptions(
                artifacts_path="/app/models",
                do_table_structure=True,
                do_ocr=True,
                ocr_options=[
                    EasyOcrOptions(),  # Default EasyOCR for general use
                    TesseractOcrOptions(
                        config='--oem 1 --psm 3 --osd'  # Enable script/orientation detection
                    )
                ]
            )
            
            # Configure document converter with format options
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
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
                'text/markdown': InputFormat.MARKDOWN,
                'text/asciidoc': InputFormat.ASCIIDOC,
                'text/html': InputFormat.HTML,
                'application/xhtml+xml': InputFormat.HTML,
                'text/csv': InputFormat.CSV,
                'image/png': InputFormat.IMAGE,
                'image/jpeg': InputFormat.IMAGE,
                'image/tiff': InputFormat.IMAGE,
                'image/bmp': InputFormat.IMAGE,
                'application/xml': InputFormat.XML,
                'application/json': InputFormat.JSON
            }

            input_format = mime_to_format.get(mime_type)
            if not input_format:
                raise ValueError(f"Unsupported file type: {mime_type}")

            # Parse document with Docling
            doc_result = converter.convert(file_path)
            
            # Extract content as markdown for better structure preservation
            text_content = doc_result.document.export_to_markdown()

            # Create document record
            async with db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO documents (id, project_id, file_name, metadata)
                    VALUES ($1, $2, $3, $4)
                """, 
                file_id, 
                inputs["project_id"], 
                inputs["file_name"],
                json.dumps({
                    "format": input_format.value,
                    "mime_type": mime_type
                })
            )

            # Initialize embeddings model
            embeddings = SentenceTransformerEmbeddings(
                "nomic-ai/nomic-embed-text-v2-moe",
                prompt_name="passage"  # Use "passage" for documents
            )

            # Initialize chunker with Nomic embeddings
            tokenizer = Tokenizer.from_pretrained("gpt2")
            chunker = SDPMChunker(
                tokenizer=tokenizer,
                chunk_size=512,  # Base chunk size
                overlap=50,      # Overlap for context preservation
                min_chunk_size=128,  # Minimum chunk size to prevent too small chunks
                merge_threshold=0.8,  # Higher threshold for better semantic coherence
                batch_size=32,    # Process chunks in batches for efficiency
                embedding_model=embeddings  # Use Nomic embeddings
            )

            # Create chunks - SDPMChunker will automatically generate embeddings
            chunks = chunker(text_content)
            
            # Prepare chunks for database insertion
            chunk_records = []
            for i, chunk in enumerate(chunks):
                chunk_records.append({
                    "content": chunk.text,
                    "token_count": chunk.token_count,
                    "document_id": file_id,
                    "project_id": inputs["project_id"],
                    "chunk_index": i,
                    "requires_embedding": False,  # Embeddings are already generated
                    "embedding": chunk.embedding,  # Use the embedding from the chunk
                    "metadata": json.dumps({
                        "format": input_format.value,
                        "chunk_type": "markdown"
                    })
                })

            # Insert chunks into database - no need for separate embedding generation
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
            prompt_name=request.prompt_name
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
