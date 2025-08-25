"""
MCP RAG Server for Document Processing and Retrieval
====================================================

A comprehensive system for extracting, embedding, and retrieving information 
from documents (particularly textbooks) to assist with coding applications.

Components:
- Document Processing: PDF text extraction with structure detection
- Embedding System: Flexible model abstraction with nomic-text-embed default
- Database: LanceDB with vector storage and metadata
- Server: FastAPI with hybrid search capabilities
- CLI: Document processing and embedding utilities
"""

__version__ = "0.1.0"
__author__ = "Drew Hayes"