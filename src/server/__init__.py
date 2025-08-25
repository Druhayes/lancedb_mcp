import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import numpy as np
from jose import JWTError, jwt

from ..config import (
    SERVER_HOST,
    SERVER_PORT,
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    LOG_LEVEL
)
from ..db import LanceDBHandler, create_db_handler
from ..embedding import EmbeddingManager, create_embedder

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# Initialize FastAPI app
app = FastAPI(
    title="MCP RAG Server",
    description="Document retrieval and embedding server for coding assistance",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for handlers
db_handler: Optional[LanceDBHandler] = None
embedding_manager: Optional[EmbeddingManager] = None
feedback_storage: List[Dict] = []  # In-memory storage for demo


# Pydantic models
class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query")
    table_name: str = Field(..., description="Table to search in")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    content_type: Optional[str] = Field(None, description="Filter by content type")


class HybridSearchQuery(SearchQuery):
    semantic_weight: float = Field(0.7, ge=0.0, le=1.0, description="Weight for semantic search")


class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    content_type: str
    page: int
    chapter_title: str
    section_title: str
    latex: str
    score: Optional[float] = None
    combined_score: Optional[float] = None
    metadata: Dict


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query: str
    total_results: int
    search_type: str


class FeedbackRequest(BaseModel):
    query: str = Field(..., description="Original search query")
    result_id: str = Field(..., description="ID of the result")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    comments: Optional[str] = Field(None, description="Optional feedback comments")


class DocumentInfo(BaseModel):
    id: str
    title: str
    source: str
    created_at: str


class HealthCheck(BaseModel):
    status: str
    timestamp: str
    db_status: str
    embedding_status: str


# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate JWT token and get current user."""
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_db_handler() -> LanceDBHandler:
    """Get database handler instance."""
    global db_handler
    if db_handler is None:
        db_handler = create_db_handler()
    return db_handler


def get_embedding_manager() -> EmbeddingManager:
    """Get embedding manager instance."""
    global embedding_manager
    if embedding_manager is None:
        embedding_manager = create_embedder()
    return embedding_manager


# Context augmentation middleware
@app.middleware("http")
async def augment_context_middleware(request: Request, call_next):
    """Middleware to augment search results for LLM context."""
    response = await call_next(request)
    
    # Add context headers for LLM integration
    if request.url.path.startswith("/search"):
        response.headers["X-Context-Type"] = "document-search"
        response.headers["X-Server-Name"] = "MCP-RAG-Server"
    
    return response


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global db_handler, embedding_manager
    
    try:
        logger.info("Starting MCP RAG Server...")
        
        # Initialize database handler
        db_handler = create_db_handler()
        logger.info("Database handler initialized")
        
        # Initialize embedding manager
        embedding_manager = create_embedder()
        logger.info("Embedding manager initialized")
        
        logger.info("MCP RAG Server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global db_handler
    
    logger.info("Shutting down MCP RAG Server...")
    
    if db_handler:
        db_handler.close()
    
    logger.info("MCP RAG Server shutdown complete")


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "MCP RAG Server",
        "version": "0.1.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    db_status = "ok"
    embedding_status = "ok"
    
    try:
        db = get_db_handler()
        # Test database connection
        db.db.table_names()
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    try:
        embedder = get_embedding_manager()
        # Test embedding
        embedder.embed_query("test")
    except Exception as e:
        embedding_status = f"error: {str(e)}"
    
    return HealthCheck(
        status="healthy" if db_status == "ok" and embedding_status == "ok" else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        db_status=db_status,
        embedding_status=embedding_status
    )


@app.post("/search/semantic", response_model=SearchResponse)
@limiter.limit("30/minute")
async def semantic_search(
    request: Request,
    search_query: SearchQuery,
    current_user: str = Depends(get_current_user)
):
    """Perform semantic search using vector similarity."""
    try:
        db = get_db_handler()
        embedder = get_embedding_manager()
        
        # Generate query embedding
        query_embedding = embedder.embed_query(search_query.query)
        
        # Perform search
        results = db.semantic_search(
            query_embedding=query_embedding,
            table_name=search_query.table_name,
            limit=search_query.limit,
            content_type=search_query.content_type
        )
        
        # Convert to response format
        search_results = [SearchResult(**result) for result in results]
        
        return SearchResponse(
            results=search_results,
            query=search_query.query,
            total_results=len(results),
            search_type="semantic"
        )
        
    except Exception as e:
        logger.error(f"Semantic search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/keyword", response_model=SearchResponse)
@limiter.limit("30/minute")
async def keyword_search(
    request: Request,
    search_query: SearchQuery,
    current_user: str = Depends(get_current_user)
):
    """Perform keyword search in document content."""
    try:
        db = get_db_handler()
        
        # Perform search
        results = db.keyword_search(
            query=search_query.query,
            table_name=search_query.table_name,
            limit=search_query.limit
        )
        
        # Convert to response format
        search_results = [SearchResult(**result) for result in results]
        
        return SearchResponse(
            results=search_results,
            query=search_query.query,
            total_results=len(results),
            search_type="keyword"
        )
        
    except Exception as e:
        logger.error(f"Keyword search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/hybrid", response_model=SearchResponse)
@limiter.limit("20/minute")
async def hybrid_search(
    request: Request,
    search_query: HybridSearchQuery,
    current_user: str = Depends(get_current_user)
):
    """Perform hybrid search combining semantic and keyword approaches."""
    try:
        db = get_db_handler()
        embedder = get_embedding_manager()
        
        # Generate query embedding
        query_embedding = embedder.embed_query(search_query.query)
        
        # Perform hybrid search
        results = db.hybrid_search(
            query=search_query.query,
            query_embedding=query_embedding,
            table_name=search_query.table_name,
            limit=search_query.limit,
            semantic_weight=search_query.semantic_weight
        )
        
        # Convert to response format
        search_results = [SearchResult(**result) for result in results]
        
        return SearchResponse(
            results=search_results,
            query=search_query.query,
            total_results=len(results),
            search_type="hybrid"
        )
        
    except Exception as e:
        logger.error(f"Hybrid search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
@limiter.limit("10/minute")
async def collect_feedback(
    request: Request,
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Collect feedback on search results for model improvement."""
    try:
        feedback_entry = {
            "id": len(feedback_storage) + 1,
            "user": current_user,
            "query": feedback.query,
            "result_id": feedback.result_id,
            "rating": feedback.rating,
            "comments": feedback.comments,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Store feedback (in production, use proper database)
        feedback_storage.append(feedback_entry)
        
        # Process feedback asynchronously
        background_tasks.add_task(process_feedback, feedback_entry)
        
        return {"message": "Feedback received", "id": feedback_entry["id"]}
        
    except Exception as e:
        logger.error(f"Feedback collection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{table_name}", response_model=List[DocumentInfo])
async def list_documents(
    table_name: str,
    current_user: str = Depends(get_current_user)
):
    """List all documents in a table."""
    try:
        db = get_db_handler()
        documents = db.list_documents(table_name)
        
        return [DocumentInfo(**doc) for doc in documents]
        
    except Exception as e:
        logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/document/{table_name}/{doc_id}")
async def get_document(
    table_name: str,
    doc_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get a specific document by ID."""
    try:
        db = get_db_handler()
        document = db.get_document_by_id(doc_id, table_name)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tables")
async def list_tables(current_user: str = Depends(get_current_user)):
    """List all available tables."""
    try:
        db = get_db_handler()
        tables = db.db.table_names()
        
        return {"tables": tables}
        
    except Exception as e:
        logger.error(f"List tables error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Admin endpoints (simplified auth for demo)
@app.get("/admin/feedback")
async def get_feedback(current_user: str = Depends(get_current_user)):
    """Get collected feedback (admin only)."""
    # In production, add proper admin role checking
    return {"feedback": feedback_storage}


@app.get("/admin/stats")
async def get_stats(current_user: str = Depends(get_current_user)):
    """Get server statistics (admin only)."""
    return {
        "total_feedback": len(feedback_storage),
        "server_uptime": "unknown",  # Could be tracked
        "total_searches": "unknown"  # Could be tracked
    }


# Background tasks
async def process_feedback(feedback_entry: Dict):
    """Process feedback for model improvement."""
    logger.info(f"Processing feedback: {feedback_entry['id']}")
    
    # Here you could implement:
    # - Update model weights based on feedback
    # - Store feedback in permanent storage
    # - Trigger retraining processes
    # - Analyze feedback patterns
    
    # For now, just log it
    logger.info(f"Feedback processed for query: {feedback_entry['query']}")


# Utility functions for JWT (simplified for demo)
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


# Development endpoint for getting token (remove in production)
@app.post("/auth/token")
async def login(username: str = "demo_user"):
    """Generate access token for demo purposes."""
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
        log_level="info"
    )