# Document Ingest Server

A FastAPI-based service for ingesting, chunking, and embedding documents with semantic search capabilities. The service processes various document formats, breaks them into meaningful chunks, and stores them in a PostgreSQL database with vector embeddings.

## Features

- Document ingestion with automatic format detection
- Support for multiple file formats (PDF, DOCX, XLSX, PPTX, text, images, etc.)
- OCR capabilities for scanned documents and images
- Semantic chunking with overlapping windows for context preservation
- Vector embeddings using Nomic AI's text embedding model
- PostgreSQL storage with pgvector for vector similarity search
- Document versioning support
- REST API for document ingestion and query embedding

## Main Technologies

- **FastAPI**: Web framework for building APIs
- **Docling**: Document processing and conversion
- **Chonkie**: Semantic document chunking
- **Sentence Transformers**: Text embeddings (nomic-ai/nomic-embed-text-v2-moe)
- **PostgreSQL + pgvector**: Vector similarity search
- **Hatchet**: Workflow orchestration
- **Docker**: Containerization and deployment

## API Usage

### Ingest a Document

```bash
curl -X POST "http://localhost:8000/ingest/file" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "project_id=my-project"
```

Response:
```json
{
  "status": "accepted",
  "file_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

### Generate Query Embeddings

```bash
curl -X POST "http://localhost:8000/embed/query" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is machine learning?",
    "prompt_name": "query"
  }'
```

Response:
```json
{
  "embedding": [...],  # 768-dimensional vector
  "model": "nomic-embed-text-v2-moe",
  "prompt_name": "query"
}
```

## Setup and Running

1. Clone the repository
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Configure the environment variables in `.env`:
   ```
   HATCHET_CLIENT_TOKEN="generate-me"
   HATCHET_CLIENT_TLS_STRATEGY=none

   # Database Configuration
   INGEST_DB_HOST=your-db-host
   INGEST_DB_PORT=your-db-port
   INGEST_DB_NAME=your-db-name
   INGEST_DB_USER=your-db-user
   INGEST_DB_PASSWORD=your-db-password
   INGEST_DB_SCHEMA=your-schema  # defaults to public
   ```

4. Start the services using Docker Compose:
   ```bash
   docker compose up -d
   ```

This will start:
- PostgreSQL for Hatchet (port 5433)
- Hatchet Lite server (ports 8888, 7077)
- Ingest API server (port 8000)

## Database Schema

The service uses two main tables:

1. `documents`: Stores document metadata and versioning information
2. `document_chunks`: Stores the actual document chunks with their embeddings

The schema includes:
- Vector similarity search using HNSW index
- Document versioning support
- Metadata storage for both documents and chunks
- Automatic timestamp management

## Development

To run the service in development mode:

1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the FastAPI server with hot reload:
   ```bash
   uvicorn main:app --reload
   ```

## License

MIT License

Copyright (c) 2025 Aikoa Intentions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
