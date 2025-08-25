import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import lancedb
from lancedb.table import Table

from ..config import (
    LANCEDB_PATH,
    VECTOR_DIMENSION,
    LOG_LEVEL
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


class LanceDBHandler:
    """
    Handler for LanceDB operations including document storage and retrieval.
    """
    
    def __init__(self, db_path: str = LANCEDB_PATH, vector_dimension: int = None):
        """
        Initialize LanceDB connection.
        
        Args:
            db_path: Path to LanceDB database
            vector_dimension: Dimension of embeddings (if None, uses config default)
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.vector_dimension = vector_dimension or VECTOR_DIMENSION
        
        try:
            self.db = lancedb.connect(str(self.db_path))
            logger.info(f"Connected to LanceDB at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {str(e)}")
            raise
    
    def create_document_table(self, table_name: str, embedding_dim: int = None) -> Table:
        """
        Create a table for storing document embeddings and metadata.
        
        Args:
            table_name: Name of the table
            embedding_dim: Dimension of embeddings
            
        Returns:
            LanceDB table instance
        """
        embedding_dim = embedding_dim or self.vector_dimension
        # Define schema for document table
        schema = {
            "id": str,
            "title": str,
            "source": str,
            "content": str,
            "content_type": str,  # 'document', 'chapter', 'section', 'equation'
            "page": int,
            "chapter_title": str,
            "section_title": str,
            "latex": str,  # For equations
            "embedding": np.ndarray,  # Vector embedding
            "metadata": str,  # JSON string for additional metadata
            "checksum": str,  # For version tracking
            "created_at": str,
            "updated_at": str
        }
        
        try:
            # Check if table already exists
            existing_tables = self.db.table_names()
            if table_name in existing_tables:
                logger.info(f"Table '{table_name}' already exists")
                return self.db.open_table(table_name)
            
            # Create sample data to initialize table
            sample_data = [{
                "id": "sample_id",
                "title": "Sample Document",
                "source": "sample.pdf",
                "content": "Sample content",
                "content_type": "document",
                "page": 1,
                "chapter_title": "",
                "section_title": "",
                "latex": "",
                "embedding": np.zeros(embedding_dim),
                "metadata": "{}",
                "checksum": "sample_checksum",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00"
            }]
            
            table = self.db.create_table(table_name, sample_data)
            
            # Delete sample data
            table.delete("id = 'sample_id'")
            
            # Note: Vector index will be created after first data insertion
            
            logger.info(f"Created table '{table_name}' with vector index")
            return table
            
        except Exception as e:
            logger.error(f"Error creating table '{table_name}': {str(e)}")
            raise
    
    def create_summary_table(self, embedding_dim: int = None) -> Table:
        """
        Create a table for storing document summaries.
        
        Args:
            embedding_dim: Dimension of embeddings
            
        Returns:
            LanceDB table instance
        """
        embedding_dim = embedding_dim or self.vector_dimension
        table_name = "document_summaries"
        
        # Define schema for summary table
        schema = {
            "document_id": str,
            "title": str,
            "source": str,
            "summary": str,
            "total_pages": int,
            "chapter_count": int,
            "section_count": int,
            "equation_count": int,
            "summary_embedding": np.ndarray,
            "metadata": str,
            "checksum": str,
            "created_at": str,
            "updated_at": str
        }
        
        try:
            existing_tables = self.db.table_names()
            if table_name in existing_tables:
                logger.info(f"Summary table already exists")
                return self.db.open_table(table_name)
            
            # Create sample data
            sample_data = [{
                "document_id": "sample_doc_id",
                "title": "Sample Document",
                "source": "sample.pdf",
                "summary": "Sample summary",
                "total_pages": 1,
                "chapter_count": 0,
                "section_count": 0,
                "equation_count": 0,
                "summary_embedding": np.zeros(embedding_dim),
                "metadata": "{}",
                "checksum": "sample_checksum",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00"
            }]
            
            table = self.db.create_table(table_name, sample_data)
            table.delete("document_id = 'sample_doc_id'")
            
            # Note: Vector index will be created after first data insertion
            
            logger.info(f"Created summary table with vector index")
            return table
            
        except Exception as e:
            logger.error(f"Error creating summary table: {str(e)}")
            raise
    
    def insert_document(self, document_data: Dict, table_name: str) -> str:
        """
        Insert a processed and embedded document into the database.
        
        Args:
            document_data: Document data with embeddings
            table_name: Name of the table to insert into
            
        Returns:
            Document ID
        """
        try:
            table = self.get_or_create_table(table_name)
            
            # Generate document ID
            doc_id = self._generate_document_id(document_data)
            
            # Prepare document entries
            entries = []
            
            # Main document entry
            if document_data.get('document_embedding'):
                embedding = self._ensure_fixed_size_embedding(document_data['document_embedding'])
                entries.append({
                    "id": f"{doc_id}_doc",
                    "title": document_data.get('title', ''),
                    "source": document_data.get('source', ''),
                    "content": document_data.get('raw_text', ''),
                    "content_type": "document",
                    "page": 1,
                    "chapter_title": "",
                    "section_title": "",
                    "latex": "",
                    "embedding": embedding,
                    "metadata": json.dumps(document_data.get('metadata', {})),
                    "checksum": self._calculate_checksum(document_data.get('raw_text', '')),
                    "created_at": pd.Timestamp.now().isoformat(),
                    "updated_at": pd.Timestamp.now().isoformat()
                })
            
            # Chapter entries
            for i, chapter in enumerate(document_data.get('chapters', [])):
                if chapter.get('embedding'):
                    embedding = self._ensure_fixed_size_embedding(chapter['embedding'])
                    entries.append({
                        "id": f"{doc_id}_ch_{i}",
                        "title": document_data.get('title', ''),
                        "source": document_data.get('source', ''),
                        "content": chapter.get('content', ''),
                        "content_type": "chapter",
                        "page": chapter.get('page', 1),
                        "chapter_title": chapter.get('title', ''),
                        "section_title": "",
                        "latex": "",
                        "embedding": embedding,
                        "metadata": json.dumps(chapter),
                        "checksum": self._calculate_checksum(chapter.get('content', '')),
                        "created_at": pd.Timestamp.now().isoformat(),
                        "updated_at": pd.Timestamp.now().isoformat()
                    })
            
            # Section entries
            for i, section in enumerate(document_data.get('sections', [])):
                if section.get('embedding'):
                    embedding = self._ensure_fixed_size_embedding(section['embedding'])
                    entries.append({
                        "id": f"{doc_id}_sec_{i}",
                        "title": document_data.get('title', ''),
                        "source": document_data.get('source', ''),
                        "content": section.get('content', ''),
                        "content_type": "section",
                        "page": section.get('page', 1),
                        "chapter_title": "",
                        "section_title": section.get('title', ''),
                        "latex": "",
                        "embedding": embedding,
                        "metadata": json.dumps(section),
                        "checksum": self._calculate_checksum(section.get('content', '')),
                        "created_at": pd.Timestamp.now().isoformat(),
                        "updated_at": pd.Timestamp.now().isoformat()
                    })
            
            # Equation entries
            for i, equation in enumerate(document_data.get('equations', [])):
                if equation.get('embedding'):
                    embedding = self._ensure_fixed_size_embedding(equation['embedding'])
                    entries.append({
                        "id": f"{doc_id}_eq_{i}",
                        "title": document_data.get('title', ''),
                        "source": document_data.get('source', ''),
                        "content": equation.get('context', ''),
                        "content_type": "equation",
                        "page": equation.get('page', 1),
                        "chapter_title": "",
                        "section_title": "",
                        "latex": equation.get('latex', ''),
                        "embedding": embedding,
                        "metadata": json.dumps(equation),
                        "checksum": self._calculate_checksum(equation.get('latex', '')),
                        "created_at": pd.Timestamp.now().isoformat(),
                        "updated_at": pd.Timestamp.now().isoformat()
                    })
            
            # Insert all entries
            if entries:
                table.add(entries)
                logger.info(f"Inserted {len(entries)} entries for document: {document_data.get('title', 'Unknown')}")
                
                # Create index if it doesn't exist and we have data
                self._ensure_index_exists(table, "embedding")
            
            # Insert summary entry
            self._insert_summary(document_data, doc_id)
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")
            raise
    
    def _insert_summary(self, document_data: Dict, doc_id: str):
        """Insert document summary into summary table."""
        try:
            summary_table = self.get_or_create_summary_table()
            
            summary_entry = {
                "document_id": doc_id,
                "title": document_data.get('title', ''),
                "source": document_data.get('source', ''),
                "summary": document_data.get('summary', ''),
                "total_pages": document_data.get('total_pages', 0),
                "chapter_count": len(document_data.get('chapters', [])),
                "section_count": len(document_data.get('sections', [])),
                "equation_count": len(document_data.get('equations', [])),
                "summary_embedding": self._ensure_fixed_size_embedding(document_data.get('summary_embedding', [])),
                "metadata": json.dumps(document_data.get('metadata', {})),
                "checksum": self._calculate_checksum(document_data.get('raw_text', '')),
                "created_at": pd.Timestamp.now().isoformat(),
                "updated_at": pd.Timestamp.now().isoformat()
            }
            
            summary_table.add([summary_entry])
            logger.info(f"Inserted summary for document: {doc_id}")
            
            # Create index if it doesn't exist and we have data
            self._ensure_index_exists(summary_table, "summary_embedding")
            
        except Exception as e:
            logger.error(f"Error inserting summary: {str(e)}")
            raise
    
    def semantic_search(self, query_embedding: np.ndarray, table_name: str, 
                       limit: int = 10, content_type: Optional[str] = None) -> List[Dict]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query_embedding: Query vector
            table_name: Table to search in
            limit: Maximum number of results
            content_type: Filter by content type
            
        Returns:
            List of search results
        """
        try:
            table = self.db.open_table(table_name)
            
            # Build search query
            search_query = table.search(query_embedding).limit(limit)
            
            # Add content type filter if specified
            if content_type:
                search_query = search_query.where(f"content_type = '{content_type}'")
            
            results = search_query.to_list()
            
            # Convert results to more usable format
            formatted_results = []
            for result in results:
                # Handle different table schemas - summary table uses 'summary' instead of 'content'
                content = result.get("content") or result.get("summary", "")
                
                formatted_results.append({
                    "id": result.get("id") or result.get("document_id"),
                    "title": result.get("title"),
                    "content": content,
                    "content_type": result.get("content_type", "summary"),
                    "page": result.get("page"),
                    "chapter_title": result.get("chapter_title"),
                    "section_title": result.get("section_title"),
                    "latex": result.get("latex"),
                    "score": result.get("_distance"),  # LanceDB returns distance
                    "metadata": json.loads(result.get("metadata", "{}"))
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise
    
    def keyword_search(self, query: str, table_name: str, limit: int = 10) -> List[Dict]:
        """
        Perform keyword search in content using SQL WHERE clause.
        
        Args:
            query: Search query
            table_name: Table to search in
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            table = self.db.open_table(table_name)
            
            # Use SQL WHERE clause for text search instead of full-text search
            # Build query based on table schema
            if table_name == "document_summaries":
                query_sql = f"summary LIKE '%{query}%' OR title LIKE '%{query}%'"
            else:
                query_sql = f"content LIKE '%{query}%' OR title LIKE '%{query}%' OR latex LIKE '%{query}%'"
            results = table.search().where(query_sql).limit(limit).to_list()
            
            formatted_results = []
            for result in results:
                # Handle different table schemas - summary table uses 'summary' instead of 'content'
                content = result.get("content") or result.get("summary", "")
                
                formatted_results.append({
                    "id": result.get("id") or result.get("document_id"),
                    "title": result.get("title"),
                    "content": content,
                    "content_type": result.get("content_type", "summary"),
                    "page": result.get("page"),
                    "chapter_title": result.get("chapter_title"),
                    "section_title": result.get("section_title"),
                    "latex": result.get("latex"),
                    "metadata": json.loads(result.get("metadata", "{}"))
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            raise
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, table_name: str,
                     limit: int = 10, semantic_weight: float = 0.7) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Text query
            query_embedding: Query vector
            table_name: Table to search in
            limit: Maximum number of results
            semantic_weight: Weight for semantic search (0-1)
            
        Returns:
            Combined and ranked search results
        """
        try:
            # Get semantic results
            semantic_results = self.semantic_search(query_embedding, table_name, limit * 2)
            
            # Get keyword results
            keyword_results = self.keyword_search(query, table_name, limit * 2)
            
            # Combine and rank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, semantic_weight
            )
            
            return combined_results[:limit]
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise
    
    def _combine_search_results(self, semantic_results: List[Dict], 
                               keyword_results: List[Dict], 
                               semantic_weight: float) -> List[Dict]:
        """Combine and rank search results from different methods."""
        result_map = {}
        
        # Add semantic results
        for i, result in enumerate(semantic_results):
            result_id = result["id"]
            semantic_score = 1.0 / (i + 1)  # Rank-based scoring
            
            result_map[result_id] = result.copy()
            result_map[result_id]["combined_score"] = semantic_score * semantic_weight
        
        # Add keyword results
        keyword_weight = 1.0 - semantic_weight
        for i, result in enumerate(keyword_results):
            result_id = result["id"]
            keyword_score = 1.0 / (i + 1)
            
            if result_id in result_map:
                result_map[result_id]["combined_score"] += keyword_score * keyword_weight
            else:
                result_map[result_id] = result.copy()
                result_map[result_id]["combined_score"] = keyword_score * keyword_weight
        
        # Sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return combined_results
    
    def get_or_create_table(self, table_name: str) -> Table:
        """Get existing table or create new one."""
        existing_tables = self.db.table_names()
        if table_name in existing_tables:
            return self.db.open_table(table_name)
        else:
            return self.create_document_table(table_name)
    
    def get_or_create_summary_table(self) -> Table:
        """Get existing summary table or create new one."""
        return self.create_summary_table()
    
    def list_documents(self, table_name: str) -> List[Dict]:
        """List all documents in a table."""
        try:
            table = self.db.open_table(table_name)
            results = table.search().where("content_type = 'document'").to_list()
            
            return [{
                "id": result.get("id"),
                "title": result.get("title"),
                "source": result.get("source"),
                "created_at": result.get("created_at")
            } for result in results]
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    def get_document_by_id(self, doc_id: str, table_name: str) -> Optional[Dict]:
        """Get a specific document by ID."""
        try:
            table = self.db.open_table(table_name)
            results = table.search().where(f"id = '{doc_id}'").to_list()
            
            if results:
                result = results[0]
                return {
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "content": result.get("content"),
                    "content_type": result.get("content_type"),
                    "metadata": json.loads(result.get("metadata", "{}"))
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document by ID: {str(e)}")
            return None
    
    def _generate_document_id(self, document_data: Dict) -> str:
        """Generate unique document ID."""
        source = document_data.get('source', '')
        title = document_data.get('title', '')
        content_hash = self._calculate_checksum(document_data.get('raw_text', ''))
        
        combined = f"{source}_{title}_{content_hash}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _ensure_index_exists(self, table: Table, column_name: str):
        """Ensure vector index exists on the given column."""
        try:
            # Check if index exists by trying to create it
            # LanceDB will raise an error if index already exists
            table.create_index(vector_column_name=column_name)
            logger.info(f"Created vector index for column: {column_name}")
        except Exception as e:
            if "already exists" in str(e).lower() or "index" in str(e).lower():
                logger.debug(f"Index for {column_name} already exists or creation skipped")
            else:
                logger.warning(f"Could not create index for {column_name}: {str(e)}")
    
    def _ensure_fixed_size_embedding(self, embedding: Union[List, np.ndarray]) -> np.ndarray:
        """Ensure embedding has the correct fixed size."""
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        if len(embedding) == 0:
            # Return zero vector of expected dimension
            return np.zeros(self.vector_dimension, dtype=np.float32)
        elif len(embedding) != self.vector_dimension:
            # Pad or truncate to expected dimension
            if len(embedding) < self.vector_dimension:
                # Pad with zeros
                padded = np.zeros(self.vector_dimension, dtype=np.float32)
                padded[:len(embedding)] = embedding
                return padded
            else:
                # Truncate
                return embedding[:self.vector_dimension].astype(np.float32)
        else:
            return embedding.astype(np.float32)
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate checksum for content versioning."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def close(self):
        """Close database connection."""
        # LanceDB doesn't require explicit closing
        logger.info("Database connection closed")


def create_db_handler(db_path: str = LANCEDB_PATH, vector_dimension: int = None) -> LanceDBHandler:
    """
    Factory function to create database handler.
    
    Args:
        db_path: Path to database
        vector_dimension: Dimension of embeddings
        
    Returns:
        LanceDBHandler instance
    """
    return LanceDBHandler(db_path, vector_dimension)
