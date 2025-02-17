import os
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from hatchet_sdk import Hatchet, Context
from dotenv import load_dotenv
import uvicorn
import asyncio
import logging
from uuid import uuid4

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Hatchet client
hatchet_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app startup and shutdown"""
    # Startup
    global hatchet_client
    logger.info("Starting up ingest service...")
    hatchet_client = Hatchet(
        api_url=os.getenv("HATCHET_API_URL", "http://localhost:8888"),
        grpc_url=os.getenv("HATCHET_GRPC_URL", "localhost:7077"),
        debug=True
    )
    
    # Start Hatchet worker
    worker = hatchet_client.worker("ingest-worker")
    worker.register_workflow(DocumentIngestWorkflow())
    worker_task = asyncio.create_task(worker.async_start())
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down ingest service...")
    worker_task.cancel()
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

        # Trigger Hatchet workflow using global client
        await hatchet_client.trigger_event("document:ingest", {
            "file_id": file_id,
            "file_path": temp_path,
            "file_name": file.filename,
            "project_id": project_id,
            "preprocess": preprocess,
            "preprocess_prompt": preprocess_prompt
        })

        return {"status": "accepted", "file_id": file_id}

    except Exception as e:
        logger.error(f"Error in file ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@hatchet_client.workflow(on_events=["document:ingest"])
class DocumentIngestWorkflow:
    @hatchet_client.step()
    async def parse_and_chunk(self, context: Context):
        """Parse document and create chunks"""
        from docling.document_converter import DocumentConverter
        from chonkie import TokenChunker
        from tokenizers import Tokenizer

        inputs = context.workflow_input()
        file_path = inputs["file_path"]
        file_id = inputs["file_id"]
        
        try:
            # Parse document with Docling
            converter = DocumentConverter()
            doc_result = converter.convert(file_path)
            
            # Extract text content
            text_content = doc_result.document.export_to_text()

            # Create document record
            async with context.get_database_connection() as conn:
                await conn.execute("""
                    INSERT INTO documents (id, project_id, file_name)
                    VALUES (:id, :project_id, :file_name)
                """, {
                    "id": file_id,
                    "project_id": inputs["project_id"],
                    "file_name": inputs["file_name"]
                })

            # Initialize chunker
            tokenizer = Tokenizer.from_pretrained("gpt2")
            chunker = TokenChunker(
                tokenizer=tokenizer,
                chunk_size=512,
                overlap=50
            )

            # Create chunks
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
                    "requires_embedding": True
                })

            # Insert chunks into database
            async with context.get_database_connection() as conn:
                await conn.executemany("""
                    INSERT INTO document_chunks (
                        content, token_count, document_id, project_id, 
                        chunk_index, requires_embedding
                    ) VALUES (
                        :content, :token_count, :document_id, :project_id,
                        :chunk_index, :requires_embedding
                    )
                """, chunk_records)

            # Clean up temporary file
            os.remove(file_path)

            return {
                "status": "success",
                "document_id": file_id,
                "chunks_created": len(chunk_records)
            }

        except Exception as e:
            logger.error(f"Error in parse_and_chunk: {e}")
            raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
