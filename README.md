# MCP RAG Server

A comprehensive document processing and retrieval system designed to assist with coding applications through intelligent document search and context augmentation.

## Features

- **Document Processing**: Extract text, structure, and equations from PDFs, text files, and Markdown
- **Flexible Embedding**: Support for multiple embedding models with easy switching
- **Vector Database**: LanceDB with efficient similarity search and hybrid retrieval
- **FastAPI Server**: RESTful API with rate limiting, authentication, and context augmentation
- **CLI Interface**: Command-line tools for document processing and server management
- **LaTeX Support**: Automatic equation detection and LaTeX formatting
- **Intelligent Structure Detection**: Automatic extraction of chapters, sections, equations, and bibliography
- **Multi-Content Search**: Search across different content types (chapters, sections, equations)
- **Document Summaries**: Automatic summary generation and searchable summary database
- **Comprehensive API**: Full REST API with authentication, feedback, and monitoring

## Architecture

```
src/
├── document_processing/   # PDF text extraction and structure detection
├── embedding/            # Flexible embedding models (transformers, sentence-transformers)
├── db/                   # LanceDB integration with vector search
├── server/               # FastAPI server with hybrid search endpoints
├── utils/                # Text processing, equation parsing, versioning utilities
├── cli.py               # Typer-based command-line interface
└── config.py            # Configuration management
```

## Complete Workflow Example

Here's a comprehensive example demonstrating the full capabilities:

```bash
# Run the complete example workflow
python example_usage.py
```

This example demonstrates:
- Document processing with structure detection
- Embedding generation for different content types
- Database storage and retrieval
- Multiple search methods (semantic, keyword, hybrid)
- Equation-specific searches
- Document management and listing
- Summary-based search

### What the Example Does

1. **Creates a sample ML textbook** with chapters, sections, equations, and bibliography
2. **Processes the document** to extract:
   - Structured content (chapters, sections)
   - LaTeX equations with proper formatting
   - Bibliography entries
   - Document metadata
3. **Generates embeddings** for all content types
4. **Stores in database** with proper indexing
5. **Demonstrates searches** across different content types
6. **Shows equation-specific search** for mathematical content
7. **Demonstrates summary search** for document discovery

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd lancedb-mcp

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env
# Edit .env with your settings
```

### 2. Run the Example Workflow

```bash
# Experience the full system capabilities
python example_usage.py
```

This will:
- Create sample data
- Process and embed documents
- Demonstrate all search types
- Show API usage examples
- Display CLI command examples

### 3. Process Your Own Documents

```bash
# Process and embed a document in one step
python main.py process-and-embed /path/to/document.pdf

# Or process in steps
python main.py process-document /path/to/document.pdf
python main.py embed-document ./data/processed/document_processed.json
```

### 4. Start the Server

```bash
# Start the MCP server
python main.py start-server

# Or with custom settings
python main.py start-server --host 0.0.0.0 --port 8080
```

### 5. Search Documents

```bash
# Get authentication token
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "demo_user"}'

# Search via CLI
python main.py search "machine learning algorithms" --type hybrid

# Or use the API
curl -X POST "http://localhost:8000/search/hybrid" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "table_name": "documents", "limit": 5}'
```

## CLI Commands

### Document Processing

```bash
# Process a single document
python main.py process-document document.pdf

# Process and embed in one step
python main.py process-and-embed document.pdf --table my_docs

# Batch process multiple documents
python main.py batch-process ./docs/ --pattern "*.pdf" --table textbooks

# Embed a processed document
python main.py embed-document processed_doc.json --table documents
```

### Database Management

```bash
# List available tables
python main.py list-tables

# List documents in a table
python main.py list-documents --table my_docs

# Search documents
python main.py search "neural networks" --type semantic --limit 10

# Search specific content types
python main.py search "backpropagation equation" --type hybrid --content-type equation

# Search with different models
python main.py search "attention mechanism" --model-type sentence_transformers
```

### Server Management

```bash
# Start server
python main.py start-server

# Start with auto-reload (development)
python main.py start-server --reload

# Start on specific host/port
python main.py start-server --host 0.0.0.0 --port 8080
```

## API Endpoints

### Authentication

**Important**: All API endpoints (except `/health`) require authentication.

```bash
# Get access token (development only)
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "demo_user"}'

# Response:
# {"access_token": "your-jwt-token", "token_type": "bearer"}
```

### Search Endpoints

```bash
# Semantic search
POST /search/semantic
{
  "query": "machine learning algorithms",
  "table_name": "documents",
  "limit": 10,
  "content_type": "chapter"  # optional filter
}

# Keyword search
POST /search/keyword
{
  "query": "neural network",
  "table_name": "documents",
  "limit": 10,
  "content_type": "section"  # optional: chapter, section, equation, document
}

# Hybrid search (recommended)
POST /search/hybrid
{
  "query": "deep learning",
  "table_name": "documents",
  "limit": 10,
  "semantic_weight": 0.7,
  "content_type": "equation"  # optional filter
}

# Equation-specific search
POST /search/semantic
{
  "query": "attention mechanism formula",
  "table_name": "ml_textbooks",
  "limit": 5,
  "content_type": "equation"
}

# Summary search for document discovery
POST /search/semantic
{
  "query": "machine learning neural networks",
  "table_name": "document_summaries",
  "limit": 3
}
```

### Document Management

```bash
# List documents
GET /documents/{table_name}

# Get specific document
GET /document/{table_name}/{doc_id}

# List tables
GET /tables

# Get document with metadata
GET /document/{table_name}/{doc_id}?include_metadata=true
```

### Feedback and Monitoring

```bash
# Submit feedback
POST /feedback
{
  "query": "original search query",
  "result_id": "document_id",
  "rating": 5,
  "comments": "Very helpful result"
}

# Health check
GET /health

# Server metrics
GET /metrics
```

## Configuration

Key configuration options in `.env`:

```bash
# Database
LANCEDB_PATH=./data/lancedb
VECTOR_DIMENSION=1024

# Embedding Model
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
MAX_BATCH_SIZE=32

# Server
SERVER_HOST=localhost
SERVER_PORT=8000
JWT_SECRET_KEY=your-secure-secret-key

# Processing
MAX_FILE_SIZE_MB=100
LOG_LEVEL=INFO
```

## Embedding Models

The system supports multiple embedding models:

### Transformers (Default)
```bash
python main.py process-and-embed doc.pdf --model-type transformers --model-name nomic-ai/nomic-embed-text-v1.5
```

### Sentence Transformers
```bash
python main.py process-and-embed doc.pdf --model-type sentence_transformers --model-name all-MiniLM-L6-v2
```

### Available Models

**Transformers Models:**
- `nomic-ai/nomic-embed-text-v1.5` (recommended, 768 dimensions)
- `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, faster)
- Any Hugging Face transformers model with embedding support

**Sentence Transformers Models:**
- `all-MiniLM-L6-v2` (fast, good quality)
- `all-mpnet-base-v2` (higher quality, slower)
- `multi-qa-MiniLM-L6-cos-v1` (optimized for Q&A)

## Document Types

### Supported Formats
- **PDF**: Full text extraction with structure detection
- **Text files**: Plain text processing with smart structure detection
- **Markdown**: Full markdown support with equation parsing

### Structure Detection

The system automatically detects and extracts:

**Document Structure:**
- Document title and metadata
- Chapter headings (multiple patterns: "Chapter 1", "1.", "# Chapter")
- Section and subsection numbering
- Table of contents (when present)
- Bibliography and reference sections

**Mathematical Content:**
- LaTeX equations (inline and display)
- Equation numbering and references
- Mathematical symbols and notation
- Formula context and explanations

**Content Organization:**
- Hierarchical section structure
- Cross-references between sections
- Contextual information for each content piece

### Equation Processing
Automatic detection and LaTeX formatting:
- Inline math: `$equation$`
- Display math: `$$equation$$`
- Equation environments: `\begin{equation}...\end{equation}`
- Align environments: `\begin{align}...\end{align}`
- AMS math environments: `\begin{gather}`, `\begin{multline}`, etc.
- Equation numbering and labeling
- Context extraction around equations

## Search Types

### Semantic Search
Uses vector embeddings for meaning-based retrieval:
```bash
python main.py search "optimization techniques" --type semantic
```

**Best for:**
- Conceptual queries
- Finding related topics
- Cross-language understanding
- Synonym matching

### Keyword Search
Traditional text-based search:
```bash
python main.py search "gradient descent" --type keyword
```

**Best for:**
- Exact term matching
- Technical terminology
- Specific names or concepts
- Fast retrieval

### Hybrid Search (Recommended)
Combines semantic and keyword approaches:
```bash
python main.py search "machine learning" --type hybrid
```

**Best for:**
- Most general queries
- Balanced precision and recall
- Unknown query types
- Production use cases

## Advanced Search Capabilities

### Content-Type Specific Search

Search within specific content types:

```bash
# Search only in equations
python main.py search "attention formula" --content-type equation

# Search only in chapters
python main.py search "neural networks" --content-type chapter

# Search only in sections
python main.py search "backpropagation" --content-type section
```

### Summary Search

Search document summaries for discovery:

```bash
# Find documents about specific topics
python main.py search "machine learning neural networks" --table document_summaries
```

### Working with Multiple Document Collections

```bash
# Create specialized collections
python main.py batch-process ./textbooks/ --table academic_books --pattern "*.pdf"
python main.py batch-process ./papers/ --table research_papers --pattern "*.pdf"
python main.py batch-process ./tutorials/ --table tutorials --pattern "*.md"

# Search across different collections
python main.py search "neural networks" --table academic_books
python main.py search "latest research" --table research_papers
python main.py search "getting started" --table tutorials
```

## Development

### Project Structure
```
├── src/
│   ├── document_processing/  # Text extraction and parsing
│   ├── embedding/           # Model abstractions
│   ├── db/                  # Database operations
│   ├── server/              # FastAPI application
│   ├── utils/               # Helper utilities
│   ├── cli.py              # Command-line interface
│   └── config.py           # Configuration
├── data/
│   ├── raw/                # Raw documents
│   ├── processed/          # Processed JSON files
│   ├── embeddings/         # Embedding cache
│   └── lancedb/            # Database files
├── example_usage.py        # Complete workflow example
├── tests/                  # Test files
├── requirements.txt        # Dependencies
├── .env.example           # Environment template
└── main.py                # Entry point
```

### Adding New Features

1. **New Document Types**: Extend `DocumentProcessor` class
2. **New Embedding Models**: Implement `EmbeddingModel` interface
3. **New Search Methods**: Add methods to `LanceDBHandler`
4. **New API Endpoints**: Add routes to `server/__init__.py`
5. **New Content Types**: Extend structure detection in document processing
6. **New Search Filters**: Add filter support in database handlers

### Testing

```bash
# Run the example workflow (comprehensive test)
python example_usage.py

# Run tests (when implemented)
python -m pytest tests/

# Test document processing
python main.py process-document test_document.pdf --verbose

# Test embedding
python main.py embed-document processed_doc.json --verbose

# Test search
python main.py search "test query" --verbose
```

### Monitoring and Debugging

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
python main.py start-server

# Check health status
curl http://localhost:8000/health

# Monitor search performance
curl -H "Authorization: Bearer <token>" http://localhost:8000/metrics
```

## Troubleshooting

### Common Issues

1. **Model Download Failures**
   - Check internet connection
   - Verify model name is correct
   - Set HF_HOME environment variable for cache location

2. **PDF Processing Errors**
   - Ensure PDF is not password protected
   - Check file permissions
   - Verify PDF is not corrupted

3. **Database Connection Issues**
   - Check LANCEDB_PATH permissions
   - Ensure sufficient disk space
   - Verify LanceDB installation

4. **Server Startup Problems**
   - Check port availability
   - Verify all dependencies installed
   - Check JWT_SECRET_KEY is set

5. **Authentication Issues**
   - Verify JWT_SECRET_KEY is set in .env
   - Check token expiration
   - Ensure proper Authorization header format

6. **Search Quality Issues**
   - Try different search types (semantic vs hybrid vs keyword)
   - Adjust semantic_weight in hybrid search
   - Use content_type filters for specific searches
   - Check if documents are properly processed and embedded

### Performance Tips

1. **Batch Processing**: Use `batch-process` for multiple documents
2. **GPU Acceleration**: Install PyTorch with CUDA support
3. **Model Selection**: Choose appropriate embedding model for your use case
4. **Index Optimization**: LanceDB automatically optimizes vector indices
5. **Content-Type Filtering**: Use specific content types to reduce search scope
6. **Embedding Caching**: Processed embeddings are cached automatically
7. **Table Organization**: Use separate tables for different document types

## Next Steps

After running the example workflow (`python example_usage.py`), you can:

1. **Start the server**: `python main.py start-server`
2. **Get an auth token**: `curl -X POST http://localhost:8000/auth/token -H "Content-Type: application/json" -d '{"username": "demo_user"}'`
3. **Search via API**: Use the token to search your documents
4. **Process your own documents**: Use the CLI commands to process your files
5. **Integrate into your applications**: Use the REST API endpoints
6. **Explore different search types**: Try semantic, keyword, and hybrid searches
7. **Browse the API docs**: Visit `http://localhost:8000/docs` when the server is running

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]

## Support

[Add support information here]
