## Me

> Context:
> - <file>mcp-build.md</file>
> - <group>full_stack_dev</group>
> - <group>neovim</group>

@{full_stack_dev}
@{neovim}

I have given you the tool access to build out the recommendations in your build instructions.  Please continue.Q

## CodeCompanion (Copilot)

I'll start building out the MCP server implementation based on our rubric. Let me begin by creating the foundational files and structure.

**`neovim__execute_command` Tool**: Returned the following:

````
Command: mkdir -p src/{document_processing,embedding,db,server,utils} data/{raw,processed,embeddings} tests
Working Directory: /home/drew-hayes/Python/lancedb-mcp
Exit Code: 0
Command completed with no output.
````


Created file `requirements.txt`
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pdfplumber==0.10.3
transformers==4.36.0
torch==2.1.0
lancedb==0.3.4
typer==0.9.0
slowapi==0.1.9
python-jose[cryptography]==3.3.0
python-multipart==0.0.6
tenacity==8.2.3
sympy==1.12
python-dotenv==1.0.0
pydantic==2.5.0
numpy==1.24.3
pandas==2.1.4
sentence-transformers==2.2.2
```


Created file `src/__init__.py`
```py
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
```


Created file `src/config.py`
```py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DATA_DIR = DATA_DIR / "embeddings"

# Database configuration
LANCEDB_PATH = os.getenv("LANCEDB_PATH", str(DATA_DIR / "lancedb"))
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "1024"))

# Model configuration
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))

# Server configuration
SERVER_HOST = os.getenv("SERVER_HOST", "localhost")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Document processing configuration
SUPPORTED_EXTENSIONS = [".pdf", ".txt", ".md"]
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```


Created file `src/document_processing/__init__.py`
```py
import logging
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pdfplumber
import sympy
from sympy.parsing.latex import parse_latex

from ..config import LOG_LEVEL, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes various document types with focus on textbooks.
    Extracts text, detects structure, and parses equations.
    """
    
    def __init__(self):
        self.chapter_patterns = [
            r'^Chapter\s+\d+',
            r'^CHAPTER\s+\d+',
            r'^\d+\.\s+[A-Z]',
            r'^Part\s+[IVX]+',
        ]
        
        self.section_patterns = [
            r'^\d+\.\d+\s+',
            r'^Section\s+\d+',
            r'^\d+\.\d+\.\d+\s+',
        ]
        
        self.equation_patterns = [
            r'\$\$([^$]+)\$\$',  # Display math
            r'\$([^$]+)\$',      # Inline math
            r'\\begin\{equation\}(.*?)\\end\{equation\}',
            r'\\begin\{align\}(.*?)\\end\{align\}',
        ]
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict:
        """
        Extract text from PDF with structure detection.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                document_data = {
                    "title": pdf_path.stem,
                    "source": str(pdf_path),
                    "total_pages": len(pdf.pages),
                    "chapters": [],
                    "sections": [],
                    "equations": [],
                    "bibliography": [],
                    "raw_text": "",
                    "metadata": {}
                }
                
                full_text = []
                current_chapter = None
                current_section = None
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    full_text.append(page_text)
                    
                    # Process each line for structure detection
                    lines = page_text.split('\n')
                    for line_num, line in enumerate(lines):
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Check for chapters
                        chapter_match = self._detect_chapter(line)
                        if chapter_match:
                            current_chapter = {
                                "title": line,
                                "page": page_num + 1,
                                "content": "",
                                "sections": []
                            }
                            document_data["chapters"].append(current_chapter)
                            logger.debug(f"Found chapter: {line}")
                        
                        # Check for sections
                        section_match = self._detect_section(line)
                        if section_match and current_chapter:
                            current_section = {
                                "title": line,
                                "page": page_num + 1,
                                "content": ""
                            }
                            current_chapter["sections"].append(current_section)
                            document_data["sections"].append(current_section)
                            logger.debug(f"Found section: {line}")
                        
                        # Extract equations
                        equations = self._extract_equations(line)
                        if equations:
                            for eq in equations:
                                document_data["equations"].append({
                                    "latex": eq,
                                    "page": page_num + 1,
                                    "context": line
                                })
                        
                        # Add content to current section/chapter
                        if current_section:
                            current_section["content"] += line + " "
                        elif current_chapter:
                            current_chapter["content"] += line + " "
                
                document_data["raw_text"] = "\n".join(full_text)
                
                # Extract bibliography if present
                document_data["bibliography"] = self._extract_bibliography(document_data["raw_text"])
                
                return document_data
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def _detect_chapter(self, line: str) -> bool:
        """Check if line contains a chapter heading."""
        for pattern in self.chapter_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _detect_section(self, line: str) -> bool:
        """Check if line contains a section heading."""
        for pattern in self.section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _extract_equations(self, text: str) -> List[str]:
        """Extract LaTeX equations from text."""
        equations = []
        
        for pattern in self.equation_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match else ""
                
                # Clean and validate equation
                cleaned_eq = self._clean_equation(match)
                if cleaned_eq and self._validate_latex(cleaned_eq):
                    equations.append(cleaned_eq)
        
        return equations
    
    def _clean_equation(self, equation: str) -> str:
        """Clean and format equation text."""
        # Remove extra whitespace
        equation = re.sub(r'\s+', ' ', equation.strip())
        
        # Basic LaTeX cleanup
        equation = equation.replace('\\\\', '\\')
        
        return equation
    
    def _validate_latex(self, equation: str) -> bool:
        """Validate LaTeX equation syntax."""
        try:
            # Try to parse with sympy
            parse_latex(equation)
            return True
        except:
            # If sympy fails, do basic validation
            return len(equation) > 2 and not equation.isspace()
    
    def _extract_bibliography(self, text: str) -> List[Dict]:
        """Extract bibliography entries from text."""
        bibliography = []
        
        # Look for bibliography section
        bib_patterns = [
            r'References\s*\n(.*?)(?=\n\s*$|\n[A-Z][A-Z\s]+\n)',
            r'Bibliography\s*\n(.*?)(?=\n\s*$|\n[A-Z][A-Z\s]+\n)',
            r'Works Cited\s*\n(.*?)(?=\n\s*$|\n[A-Z][A-Z\s]+\n)',
        ]
        
        for pattern in bib_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                bib_text = match.group(1)
                # Split into individual entries (basic approach)
                entries = re.split(r'\n\s*\n', bib_text)
                
                for i, entry in enumerate(entries):
                    if entry.strip():
                        bibliography.append({
                            "id": f"ref_{i+1}",
                            "text": entry.strip(),
                            "type": "unknown"  # Could be enhanced with citation parsing
                        })
                break
        
        return bibliography
    
    def process_document(self, file_path: Path) -> Dict:
        """
        Process a document based on its type.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Processed document data
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif suffix in ['.txt', '.md']:
            return self._process_text_file(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _process_text_file(self, file_path: Path) -> Dict:
        """Process plain text or markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            document_data = {
                "title": file_path.stem,
                "source": str(file_path),
                "total_pages": 1,
                "chapters": [],
                "sections": [],
                "equations": [],
                "bibliography": [],
                "raw_text": content,
                "metadata": {"file_type": file_path.suffix}
            }
            
            # Extract equations from text
            lines = content.split('\n')
            for line_num, line in enumerate(lines):
                equations = self._extract_equations(line)
                if equations:
                    for eq in equations:
                        document_data["equations"].append({
                            "latex": eq,
                            "page": 1,
                            "line": line_num + 1,
                            "context": line
                        })
            
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {str(e)}")
            raise
    
    def save_processed_document(self, document_data: Dict, output_path: Optional[Path] = None) -> Path:
        """
        Save processed document data to JSON.
        
        Args:
            document_data: Processed document data
            output_path: Optional custom output path
            
        Returns:
            Path to saved file
        """
        if output_path is None:
            filename = f"{document_data['title']}_processed.json"
            output_path = PROCESSED_DATA_DIR / filename
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed document to {output_path}")
        return output_path


def main():
    """CLI entry point for document processing."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m src.document_processing.processor <file_path>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    processor = DocumentProcessor()
    
    try:
        document_data = processor.process_document(file_path)
        output_path = processor.save_processed_document(document_data)
        print(f"Successfully processed document: {output_path}")
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```


## Me

@{full_stack_dev}
@{neovim}

> Context:
> - <file>mcp-build.md</file>
> - <file>build-progress.md</file>
> - <group>full_stack_dev</group>
> - <group>neovim</group>

Let's continue where we left off.  THe progress that we have made is in the build-progress.md file.

Additionally, when you call the tool to create file you must make the call for each file that you create.  You cannot send instructions to create multiple files at the same time.


## CodeCompanion (Copilot)

I'll continue building the MCP server implementation. Let me check the current progress and then create the remaining components.

Read file `build-progress.md`


I can see the progress so far. Let me continue building the embedding system next. I'll create the embedding module with the flexible model abstraction layer.

Created file `src/embedding/__init__.py`
```py
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
import torch
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import (
    DEFAULT_EMBEDDING_MODEL, 
    MAX_BATCH_SIZE, 
    VECTOR_DIMENSION,
    LOG_LEVEL
)

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Abstract base class for embedding models to ensure consistency.
    """
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts and return embeddings."""
        raise NotImplementedError
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text and return embedding."""
        raise NotImplementedError


class TransformersEmbedder(EmbeddingModel):
    """
    HuggingFace Transformers-based embedder with flexible model support.
    """
    
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: Optional[str] = None):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the HuggingFace model
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            with torch.no_grad():
                sample_input = self.tokenizer("test", return_tensors="pt", truncation=True, padding=True)
                sample_input = {k: v.to(self.device) for k, v in sample_input.items()}
                sample_output = self.model(**sample_input)
                self.embedding_dim = sample_output.last_hidden_state.mean(dim=1).shape[-1]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts with batch processing and retry logic.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch_texts = texts[i:i + MAX_BATCH_SIZE]
            batch_embeddings = self._embed_batch(batch_texts)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts."""
        try:
            # Tokenize texts
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                return embeddings.cpu().numpy()
                
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            raise
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]


class SentenceTransformersEmbedder(EmbeddingModel):
    """
    Sentence Transformers-based embedder for alternative models.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize the sentence transformer embedder.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"SentenceTransformer loaded. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer {model_name}: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=MAX_BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in SentenceTransformer embedding: {str(e)}")
            raise
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text."""
        return self.embed([text])[0]


class EmbeddingManager:
    """
    Manager class for handling different embedding models and document processing.
    """
    
    def __init__(self, model_type: str = "transformers", model_name: Optional[str] = None):
        """
        Initialize the embedding manager.
        
        Args:
            model_type: Type of embedder ('transformers' or 'sentence_transformers')
            model_name: Specific model name (uses default if None)
        """
        self.model_type = model_type
        
        if model_type == "transformers":
            model_name = model_name or DEFAULT_EMBEDDING_MODEL
            self.embedder = TransformersEmbedder(model_name)
        elif model_type == "sentence_transformers":
            model_name = model_name or "all-MiniLM-L6-v2"
            self.embedder = SentenceTransformersEmbedder(model_name)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"EmbeddingManager initialized with {model_type} model: {model_name}")
    
    def embed_document(self, document_data: Dict) -> Dict:
        """
        Embed a processed document and return enhanced data with embeddings.
        
        Args:
            document_data: Processed document data from DocumentProcessor
            
        Returns:
            Document data enhanced with embeddings
        """
        logger.info(f"Embedding document: {document_data.get('title', 'Unknown')}")
        
        enhanced_data = document_data.copy()
        
        # Embed full document text
        if document_data.get('raw_text'):
            enhanced_data['document_embedding'] = self.embedder.embed_single(
                document_data['raw_text']
            ).tolist()
        
        # Embed chapters
        if document_data.get('chapters'):
            for chapter in enhanced_data['chapters']:
                if chapter.get('content'):
                    chapter['embedding'] = self.embedder.embed_single(
                        chapter['content']
                    ).tolist()
        
        # Embed sections
        if document_data.get('sections'):
            for section in enhanced_data['sections']:
                if section.get('content'):
                    section['embedding'] = self.embedder.embed_single(
                        section['content']
                    ).tolist()
        
        # Embed equations (context)
        if document_data.get('equations'):
            equation_contexts = [eq.get('context', '') for eq in document_data['equations']]
            if equation_contexts:
                equation_embeddings = self.embedder.embed(equation_contexts)
                for i, equation in enumerate(enhanced_data['equations']):
                    equation['embedding'] = equation_embeddings[i].tolist()
        
        # Create document summary embedding
        summary_parts = []
        if document_data.get('title'):
            summary_parts.append(f"Title: {document_data['title']}")
        
        if document_data.get('chapters'):
            chapter_titles = [ch.get('title', '') for ch in document_data['chapters']]
            summary_parts.append(f"Chapters: {', '.join(chapter_titles)}")
        
        if document_data.get('equations'):
            summary_parts.append(f"Contains {len(document_data['equations'])} equations")
        
        summary_text = '. '.join(summary_parts)
        enhanced_data['summary'] = summary_text
        enhanced_data['summary_embedding'] = self.embedder.embed_single(summary_text).tolist()
        
        logger.info(f"Document embedding completed for: {document_data.get('title', 'Unknown')}")
        return enhanced_data
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts."""
        return self.embedder.embed(texts)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query for search."""
        return self.embedder.embed_single(query)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        return self.embedder.embedding_dim


def create_embedder(model_type: str = "transformers", model_name: Optional[str] = None) -> EmbeddingManager:
    """
    Factory function to create embedding manager.
    
    Args:
        model_type: Type of embedder
        model_name: Specific model name
        
    Returns:
        EmbeddingManager instance
    """
    return EmbeddingManager(model_type=model_type, model_name=model_name)
```


Now let me create the database module for LanceDB integration:

Created file `src/db/__init__.py`
```py
import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import lancedb
from lancedb.table import Table
from lancedb.index import IndexConfig, Index

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
    
    def __init__(self, db_path: str = LANCEDB_PATH):
        """
        Initialize LanceDB connection.
        
        Args:
            db_path: Path to LanceDB database
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        try:
            self.db = lancedb.connect(str(self.db_path))
            logger.info(f"Connected to LanceDB at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {str(e)}")
            raise
    
    def create_document_table(self, table_name: str, embedding_dim: int = VECTOR_DIMENSION) -> Table:
        """
        Create a table for storing document embeddings and metadata.
        
        Args:
            table_name: Name of the table
            embedding_dim: Dimension of embeddings
            
        Returns:
            LanceDB table instance
        """
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
            
            # Create vector index for efficient similarity search
            table.create_index("embedding", config=IndexConfig.ivf_pq())
            
            logger.info(f"Created table '{table_name}' with vector index")
            return table
            
        except Exception as e:
            logger.error(f"Error creating table '{table_name}': {str(e)}")
            raise
    
    def create_summary_table(self, embedding_dim: int = VECTOR_DIMENSION) -> Table:
        """
        Create a table for storing document summaries.
        
        Args:
            embedding_dim: Dimension of embeddings
            
        Returns:
            LanceDB table instance
        """
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
            
            # Create vector index
            table.create_index("summary_embedding", config=IndexConfig.ivf_pq())
            
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
                    "embedding": np.array(document_data['document_embedding']),
                    "metadata": json.dumps(document_data.get('metadata', {})),
                    "checksum": self._calculate_checksum(document_data.get('raw_text', '')),
                    "created_at": pd.Timestamp.now().isoformat(),
                    "updated_at": pd.Timestamp.now().isoformat()
                })
            
            # Chapter entries
            for i, chapter in enumerate(document_data.get('chapters', [])):
                if chapter.get('embedding'):
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
                        "embedding": np.array(chapter['embedding']),
                        "metadata": json.dumps(chapter),
                        "checksum": self._calculate_checksum(chapter.get('content', '')),
                        "created_at": pd.Timestamp.now().isoformat(),
                        "updated_at": pd.Timestamp.now().isoformat()
                    })
            
            # Section entries
            for i, section in enumerate(document_data.get('sections', [])):
                if section.get('embedding'):
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
                        "embedding": np.array(section['embedding']),
                        "metadata": json.dumps(section),
                        "checksum": self._calculate_checksum(section.get('content', '')),
                        "created_at": pd.Timestamp.now().isoformat(),
                        "updated_at": pd.Timestamp.now().isoformat()
                    })
            
            # Equation entries
            for i, equation in enumerate(document_data.get('equations', [])):
                if equation.get('embedding'):
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
                        "embedding": np.array(equation['embedding']),
                        "metadata": json.dumps(equation),
                        "checksum": self._calculate_checksum(equation.get('latex', '')),
                        "created_at": pd.Timestamp.now().isoformat(),
                        "updated_at": pd.Timestamp.now().isoformat()
                    })
            
            # Insert all entries
            if entries:
                table.add(entries)
                logger.info(f"Inserted {len(entries)} entries for document: {document_data.get('title', 'Unknown')}")
            
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
                "summary_embedding": np.array(document_data.get('summary_embedding', [])),
                "metadata": json.dumps(document_data.get('metadata', {})),
                "checksum": self._calculate_checksum(document_data.get('raw_text', '')),
                "created_at": pd.Timestamp.now().isoformat(),
                "updated_at": pd.Timestamp.now().isoformat()
            }
            
            summary_table.add([summary_entry])
            logger.info(f"Inserted summary for document: {doc_id}")
            
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
                formatted_results.append({
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "content": result.get("content"),
                    "content_type": result.get("content_type"),
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
        Perform keyword search in content.
        
        Args:
            query: Search query
            table_name: Table to search in
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            table = self.db.open_table(table_name)
            
            # Use SQL-like query for text search
            results = table.search(query).limit(limit).to_list()
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "content": result.get("content"),
                    "content_type": result.get("content_type"),
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
    
    def _calculate_checksum(self, content: str) -> str:
        """Calculate checksum for content versioning."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def close(self):
        """Close database connection."""
        # LanceDB doesn't require explicit closing
        logger.info("Database connection closed")


def create_db_handler(db_path: str = LANCEDB_PATH) -> LanceDBHandler:
    """
    Factory function to create database handler.
    
    Args:
        db_path: Path to database
        
    Returns:
        LanceDBHandler instance
    """
    return LanceDBHandler(db_path)
```


Now let me create the server module with FastAPI endpoints:

Created file `src/server/__init__.py`
```py
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
```


Now let me create the utilities module:

Created file `src/utils/__init__.py`
```py
import logging
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np

from ..config import LOG_LEVEL

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


class TextCleaner:
    """Utility class for cleaning and preprocessing text."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep equations
        # (preserve LaTeX equations)
        text = re.sub(r'[^\w\s\$\\\{\}\(\)\[\]\.,;:!?\-\+\=\<\>]', '', text)
        
        # Remove extra periods and spaces
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (could be enhanced with nltk)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into chunks with overlap for better embedding.
        
        Args:
            text: Text to chunk
            max_length: Maximum length per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            
            # Find a good break point (end of sentence)
            if end < len(text):
                # Look for sentence endings near the max length
                for i in range(end, max(start + max_length // 2, end - 100), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + 1, end - overlap)
            
            if start >= len(text):
                break
        
        return chunks


class EquationParser:
    """Utility class for parsing and validating LaTeX equations."""
    
    # Common LaTeX equation patterns
    EQUATION_PATTERNS = [
        r'\$\$([^$]+)\$\$',  # Display math
        r'\$([^$]+)\$',      # Inline math
        r'\\begin\{equation\}(.*?)\\end\{equation\}',
        r'\\begin\{align\}(.*?)\\end\{align\}',
        r'\\begin\{eqnarray\}(.*?)\\end\{eqnarray\}',
        r'\\begin\{gather\}(.*?)\\end\{gather\}',
        r'\\begin\{split\}(.*?)\\end\{split\}',
        r'\\begin\{multline\}(.*?)\\end\{multline\}',
    ]
    
    @classmethod
    def extract_equations(cls, text: str) -> List[Dict[str, str]]:
        """
        Extract LaTeX equations from text.
        
        Args:
            text: Text containing equations
            
        Returns:
            List of equation dictionaries
        """
        equations = []
        
        for i, pattern in enumerate(cls.EQUATION_PATTERNS):
            matches = re.finditer(pattern, text, re.DOTALL)
            
            for match in matches:
                equation_text = match.group(1) if len(match.groups()) > 0 else match.group(0)
                
                # Clean the equation
                cleaned_eq = cls.clean_equation(equation_text)
                
                if cleaned_eq and cls.validate_equation(cleaned_eq):
                    equations.append({
                        'latex': cleaned_eq,
                        'original': match.group(0),
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'type': cls._get_equation_type(pattern)
                    })
        
        return equations
    
    @staticmethod
    def clean_equation(equation: str) -> str:
        """
        Clean LaTeX equation string.
        
        Args:
            equation: Raw equation string
            
        Returns:
            Cleaned equation
        """
        # Remove extra whitespace
        equation = re.sub(r'\s+', ' ', equation.strip())
        
        # Fix common LaTeX issues
        equation = equation.replace('\\\\', '\\')
        equation = re.sub(r'\\([a-zA-Z]+)', r'\\\1', equation)  # Ensure backslashes
        
        return equation
    
    @staticmethod
    def validate_equation(equation: str) -> bool:
        """
        Validate LaTeX equation syntax (basic validation).
        
        Args:
            equation: LaTeX equation string
            
        Returns:
            True if valid, False otherwise
        """
        if not equation or len(equation) < 2:
            return False
        
        # Check for balanced braces
        brace_count = equation.count('{') - equation.count('}')
        if brace_count != 0:
            return False
        
        # Check for balanced parentheses
        paren_count = equation.count('(') - equation.count(')')
        if paren_count != 0:
            return False
        
        # Check for basic LaTeX commands
        has_latex_commands = bool(re.search(r'\\[a-zA-Z]+', equation))
        has_math_symbols = bool(re.search(r'[+\-*/=<>^_{}]', equation))
        
        return has_latex_commands or has_math_symbols
    
    @staticmethod
    def _get_equation_type(pattern: str) -> str:
        """Determine equation type from pattern."""
        if 'equation' in pattern:
            return 'equation'
        elif 'align' in pattern:
            return 'align'
        elif '$$' in pattern:
            return 'display'
        elif '$' in pattern:
            return 'inline'
        else:
            return 'unknown'


class MetadataExtractor:
    """Utility class for extracting metadata from documents."""
    
    @staticmethod
    def extract_title(text: str, filename: str = "") -> str:
        """
        Extract document title from text or filename.
        
        Args:
            text: Document text
            filename: Document filename
            
        Returns:
            Extracted title
        """
        # Try to find title in first few lines
        lines = text.split('\n')[:10]
        
        for line in lines:
            line = line.strip()
            
            # Look for title patterns
            title_patterns = [
                r'^Title:\s*(.+)$',
                r'^TITLE:\s*(.+)$',
                r'^#\s*(.+)$',  # Markdown title
                r'^\*\*(.+)\*\*$',  # Bold title
            ]
            
            for pattern in title_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            # If line is short and looks like a title
            if len(line) < 100 and len(line) > 5 and not line.endswith('.'):
                # Check if it's likely a title (capitalized, no lowercase articles at start)
                if line[0].isupper() and not re.match(r'^(the|a|an|of|in|on|at|to|for)\s', line, re.IGNORECASE):
                    return line
        
        # Fall back to filename
        if filename:
            return Path(filename).stem.replace('_', ' ').replace('-', ' ').title()
        
        return "Unknown Document"
    
    @staticmethod
    def extract_authors(text: str) -> List[str]:
        """
        Extract author names from text.
        
        Args:
            text: Document text
            
        Returns:
            List of author names
        """
        authors = []
        lines = text.split('\n')[:20]  # Check first 20 lines
        
        for line in lines:
            line = line.strip()
            
            # Look for author patterns
            author_patterns = [
                r'^Author[s]?:\s*(.+)$',
                r'^By:\s*(.+)$',
                r'^Written by:\s*(.+)$',
            ]
            
            for pattern in author_patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    author_text = match.group(1)
                    # Split multiple authors
                    authors.extend([a.strip() for a in re.split(r'[,&]|and', author_text)])
                    break
        
        return [a for a in authors if a]  # Remove empty strings
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from text (simple implementation).
        
        Args:
            text: Document text
            max_keywords: Maximum number of keywords
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction based on word frequency
        # In production, could use more sophisticated NLP
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in sorted_words[:max_keywords]]


class DocumentVersioning:
    """Utility class for document versioning and change tracking."""
    
    @staticmethod
    def calculate_checksum(content: str, algorithm: str = 'sha256') -> str:
        """
        Calculate checksum for content.
        
        Args:
            content: Content to hash
            algorithm: Hash algorithm to use
            
        Returns:
            Hexadecimal hash string
        """
        if algorithm == 'md5':
            return hashlib.md5(content.encode()).hexdigest()
        elif algorithm == 'sha1':
            return hashlib.sha1(content.encode()).hexdigest()
        elif algorithm == 'sha256':
            return hashlib.sha256(content.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    @staticmethod
    def generate_document_id(title: str, source: str, content_hash: str) -> str:
        """
        Generate unique document ID.
        
        Args:
            title: Document title
            source: Document source path
            content_hash: Content checksum
            
        Returns:
            Unique document ID
        """
        combined = f"{title}_{source}_{content_hash}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    @staticmethod
    def compare_documents(doc1: Dict, doc2: Dict) -> Dict[str, Any]:
        """
        Compare two document versions.
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Comparison results
        """
        comparison = {
            'title_changed': doc1.get('title') != doc2.get('title'),
            'content_changed': doc1.get('raw_text') != doc2.get('raw_text'),
            'structure_changed': False,
            'equations_changed': False
        }
        
        # Compare structure (chapters/sections)
        ch1 = [ch.get('title', '') for ch in doc1.get('chapters', [])]
        ch2 = [ch.get('title', '') for ch in doc2.get('chapters', [])]
        comparison['structure_changed'] = ch1 != ch2
        
        # Compare equations
        eq1 = [eq.get('latex', '') for eq in doc1.get('equations', [])]
        eq2 = [eq.get('latex', '') for eq in doc2.get('equations', [])]
        comparison['equations_changed'] = eq1 != eq2
        
        return comparison


class Logger:
    """Enhanced logging utility."""
    
    @staticmethod
    def setup_logger(name: str, log_file: Optional[Path] = None, level: str = LOG_LEVEL) -> logging.Logger:
        """
        Set up a logger with file and console handlers.
        
        Args:
            name: Logger name
            log_file: Optional log file path
            level: Logging level
            
        Returns:
            Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    @staticmethod
    def log_performance(func_name: str, execution_time: float, details: Optional[Dict] = None):
        """Log performance metrics."""
        logger = logging.getLogger("performance")
        
        message = f"Function '{func_name}' executed in {execution_time:.2f}s"
        if details:
            message += f" - Details: {json.dumps(details)}"
        
        logger.info(message)


class ArrayUtils:
    """Utility functions for working with numpy arrays and embeddings."""
    
    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector to unit length.
        
        Args:
            embedding: Input embedding vector
            
        Returns:
            Normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    @staticmethod
    def batch_cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarities between query and batch of vectors.
        
        Args:
            query: Query vector
            vectors: Batch of vectors
            
        Returns:
            Array of similarity scores
        """
        # Normalize vectors
        query_norm = query / np.linalg.norm(query)
        vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Calculate similarities
        similarities = np.dot(vectors_norm, query_norm)
        
        return similarities


# Export main utilities
__all__ = [
    'TextCleaner',
    'EquationParser', 
    'MetadataExtractor',
    'DocumentVersioning',
    'Logger',
    'ArrayUtils'
]
```


Now let me create the CLI interface with typer:

Created file `src/cli.py`
```py
#!/usr/bin/env python3
"""
MCP RAG Server CLI
==================

Command-line interface for processing documents and managing the MCP RAG server.
"""

import logging
import json
import sys
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

# Import our modules
from .document_processing import DocumentProcessor
from .embedding import EmbeddingManager, create_embedder
from .db import LanceDBHandler, create_db_handler
from .config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EMBEDDINGS_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    LOG_LEVEL
)

# Setup
console = Console()
app = typer.Typer(help="MCP RAG Server CLI for document processing and management")

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL))
logger = logging.getLogger(__name__)


@app.command()
def process_document(
    file_path: Path = typer.Argument(..., help="Path to document file"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Process a document (PDF, TXT, MD) and extract structured content.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate input file
        if not file_path.exists():
            rprint(f"[red]Error: File not found: {file_path}[/red]")
            raise typer.Exit(1)
        
        if not file_path.suffix.lower() in ['.pdf', '.txt', '.md']:
            rprint(f"[red]Error: Unsupported file type: {file_path.suffix}[/red]")
            raise typer.Exit(1)
        
        # Initialize processor
        processor = DocumentProcessor()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Process document
            progress.add_task("Processing document...", total=None)
            document_data = processor.process_document(file_path)
            
            # Save processed data
            if output_dir:
                output_path = output_dir / f"{document_data['title']}_processed.json"
            else:
                output_path = None
            
            progress.add_task("Saving processed data...", total=None)
            saved_path = processor.save_processed_document(document_data, output_path)
        
        # Display summary
        _display_processing_summary(document_data, saved_path)
        
        rprint(f"[green]✓ Document processed successfully![/green]")
        rprint(f"Output saved to: {saved_path}")
        
    except Exception as e:
        rprint(f"[red]Error processing document: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def embed_document(
    processed_file: Path = typer.Argument(..., help="Path to processed document JSON"),
    table_name: str = typer.Option("documents", "--table", "-t", help="Database table name"),
    model_type: str = typer.Option("transformers", "--model-type", help="Embedding model type"),
    model_name: Optional[str] = typer.Option(None, "--model-name", help="Specific model name"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Generate embeddings for a processed document and store in database.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate input file
        if not processed_file.exists():
            rprint(f"[red]Error: File not found: {processed_file}[/red]")
            raise typer.Exit(1)
        
        # Load processed document
        with open(processed_file, 'r', encoding='utf-8') as f:
            document_data = json.load(f)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Initialize embedding manager
            progress.add_task("Loading embedding model...", total=None)
            embedding_manager = create_embedder(model_type=model_type, model_name=model_name)
            
            # Generate embeddings
            progress.add_task("Generating embeddings...", total=None)
            embedded_data = embedding_manager.embed_document(document_data)
            
            # Initialize database
            progress.add_task("Connecting to database...", total=None)
            db_handler = create_db_handler()
            
            # Store in database
            progress.add_task("Storing in database...", total=None)
            doc_id = db_handler.insert_document(embedded_data, table_name)
        
        # Display summary
        _display_embedding_summary(embedded_data, doc_id, table_name)
        
        rprint(f"[green]✓ Document embedded and stored successfully![/green]")
        rprint(f"Document ID: {doc_id}")
        rprint(f"Table: {table_name}")
        
    except Exception as e:
        rprint(f"[red]Error embedding document: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def process_and_embed(
    file_path: Path = typer.Argument(..., help="Path to document file"),
    table_name: str = typer.Option("documents", "--table", "-t", help="Database table name"),
    model_type: str = typer.Option("transformers", "--model-type", help="Embedding model type"),
    model_name: Optional[str] = typer.Option(None, "--model-name", help="Specific model name"),
    keep_processed: bool = typer.Option(False, "--keep-processed", help="Keep processed JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Process and embed a document in one step.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validate input
        if not file_path.exists():
            rprint(f"[red]Error: File not found: {file_path}[/red]")
            raise typer.Exit(1)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Process document
            progress.add_task("Processing document...", total=None)
            processor = DocumentProcessor()
            document_data = processor.process_document(file_path)
            
            # Generate embeddings
            progress.add_task("Loading embedding model...", total=None)
            embedding_manager = create_embedder(model_type=model_type, model_name=model_name)
            
            progress.add_task("Generating embeddings...", total=None)
            embedded_data = embedding_manager.embed_document(document_data)
            
            # Store in database
            progress.add_task("Storing in database...", total=None)
            db_handler = create_db_handler()
            doc_id = db_handler.insert_document(embedded_data, table_name)
            
            # Save processed data if requested
            if keep_processed:
                progress.add_task("Saving processed data...", total=None)
                saved_path = processor.save_processed_document(document_data)
                rprint(f"Processed data saved to: {saved_path}")
        
        # Display summary
        _display_processing_summary(document_data)
        _display_embedding_summary(embedded_data, doc_id, table_name)
        
        rprint(f"[green]✓ Document processed, embedded, and stored successfully![/green]")
        rprint(f"Document ID: {doc_id}")
        
    except Exception as e:
        rprint(f"[red]Error processing and embedding document: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def batch_process(
    input_dir: Path = typer.Argument(..., help="Directory containing documents"),
    table_name: str = typer.Option("documents", "--table", "-t", help="Database table name"),
    model_type: str = typer.Option("transformers", "--model-type", help="Embedding model type"),
    model_name: Optional[str] = typer.Option(None, "--model-name", help="Specific model name"),
    pattern: str = typer.Option("*.pdf", "--pattern", "-p", help="File pattern to match"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Process multiple documents in a directory.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Find files
        files = list(input_dir.glob(pattern))
        if not files:
            rprint(f"[yellow]No files found matching pattern: {pattern}[/yellow]")
            return
        
        rprint(f"Found {len(files)} files to process")
        
        # Initialize components once
        processor = DocumentProcessor()
        embedding_manager = create_embedder(model_type=model_type, model_name=model_name)
        db_handler = create_db_handler()
        
        successful = 0
        failed = 0
        
        with Progress(console=console) as progress:
            task = progress.add_task("Processing documents...", total=len(files))
            
            for file_path in files:
                try:
                    progress.update(task, description=f"Processing {file_path.name}...")
                    
                    # Process document
                    document_data = processor.process_document(file_path)
                    
                    # Generate embeddings
                    embedded_data = embedding_manager.embed_document(document_data)
                    
                    # Store in database
                    doc_id = db_handler.insert_document(embedded_data, table_name)
                    
                    successful += 1
                    rprint(f"[green]✓[/green] {file_path.name} -> {doc_id}")
                    
                except Exception as e:
                    failed += 1
                    rprint(f"[red]✗[/red] {file_path.name}: {str(e)}")
                    if verbose:
                        console.print_exception()
                
                progress.advance(task)
        
        # Summary
        rprint(f"\n[green]Batch processing complete![/green]")
        rprint(f"Successful: {successful}")
        rprint(f"Failed: {failed}")
        
    except Exception as e:
        rprint(f"[red]Error in batch processing: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    table_name: str = typer.Option("documents", "--table", "-t", help="Table to search"),
    search_type: str = typer.Option("hybrid", "--type", help="Search type: semantic, keyword, hybrid"),
    limit: int = typer.Option(5, "--limit", "-l", help="Number of results"),
    model_type: str = typer.Option("transformers", "--model-type", help="Embedding model type"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Search documents in the database.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize components
        db_handler = create_db_handler()
        
        if search_type in ["semantic", "hybrid"]:
            embedding_manager = create_embedder(model_type=model_type)
            query_embedding = embedding_manager.embed_query(query)
        
        # Perform search
        if search_type == "semantic":
            results = db_handler.semantic_search(query_embedding, table_name, limit)
        elif search_type == "keyword":
            results = db_handler.keyword_search(query, table_name, limit)
        elif search_type == "hybrid":
            results = db_handler.hybrid_search(query, query_embedding, table_name, limit)
        else:
            rprint(f"[red]Invalid search type: {search_type}[/red]")
            raise typer.Exit(1)
        
        # Display results
        _display_search_results(query, results, search_type)
        
    except Exception as e:
        rprint(f"[red]Error searching: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def list_documents(
    table_name: str = typer.Option("documents", "--table", "-t", help="Table name"),
):
    """
    List all documents in a table.
    """
    try:
        db_handler = create_db_handler()
        documents = db_handler.list_documents(table_name)
        
        if not documents:
            rprint(f"[yellow]No documents found in table: {table_name}[/yellow]")
            return
        
        # Create table
        table = Table(title=f"Documents in table: {table_name}")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="magenta")
        table.add_column("Source", style="green")
        table.add_column("Created", style="yellow")
        
        for doc in documents:
            table.add_row(
                doc.get('id', ''),
                doc.get('title', ''),
                doc.get('source', ''),
                doc.get('created_at', '')
            )
        
        console.print(table)
        
    except Exception as e:
        rprint(f"[red]Error listing documents: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def list_tables():
    """
    List all tables in the database.
    """
    try:
        db_handler = create_db_handler()
        tables = db_handler.db.table_names()
        
        if not tables:
            rprint("[yellow]No tables found in database[/yellow]")
            return
        
        rprint("[green]Available tables:[/green]")
        for table in tables:
            rprint(f"  • {table}")
        
    except Exception as e:
        rprint(f"[red]Error listing tables: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command()
def start_server(
    host: str = typer.Option("localhost", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """
    Start the MCP RAG server.
    """
    try:
        import uvicorn
        from .server import app as server_app
        
        rprint(f"[green]Starting MCP RAG Server on {host}:{port}[/green]")
        
        uvicorn.run(
            server_app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except ImportError:
        rprint("[red]uvicorn not installed. Please install with: pip install uvicorn[/red]")
        raise typer.Exit(1)
    except Exception as e:
        rprint(f"[red]Error starting server: {str(e)}[/red]")
        raise typer.Exit(1)


# Helper functions for display
def _display_processing_summary(document_data: dict, saved_path: Path = None):
    """Display processing summary."""
    panel_content = f"""
[bold]Title:[/bold] {document_data.get('title', 'Unknown')}
[bold]Source:[/bold] {document_data.get('source', 'Unknown')}
[bold]Pages:[/bold] {document_data.get('total_pages', 0)}
[bold]Chapters:[/bold] {len(document_data.get('chapters', []))}
[bold]Sections:[/bold] {len(document_data.get('sections', []))}
[bold]Equations:[/bold] {len(document_data.get('equations', []))}
[bold]Bibliography entries:[/bold] {len(document_data.get('bibliography', []))}
"""
    
    if saved_path:
        panel_content += f"[bold]Saved to:[/bold] {saved_path}"
    
    console.print(Panel(panel_content, title="Processing Summary", expand=False))


def _display_embedding_summary(embedded_data: dict, doc_id: str, table_name: str):
    """Display embedding summary."""
    panel_content = f"""
[bold]Document ID:[/bold] {doc_id}
[bold]Table:[/bold] {table_name}
[bold]Document embedding:[/bold] {'✓' if embedded_data.get('document_embedding') else '✗'}
[bold]Chapter embeddings:[/bold] {sum(1 for ch in embedded_data.get('chapters', []) if ch.get('embedding'))}
[bold]Section embeddings:[/bold] {sum(1 for sec in embedded_data.get('sections', []) if sec.get('embedding'))}
[bold]Equation embeddings:[/bold] {sum(1 for eq in embedded_data.get('equations', []) if eq.get('embedding'))}
[bold]Summary embedding:[/bold] {'✓' if embedded_data.get('summary_embedding') else '✗'}
"""
    
    console.print(Panel(panel_content, title="Embedding Summary", expand=False))


def _display_search_results(query: str, results: List[dict], search_type: str):
    """Display search results."""
    if not results:
        rprint(f"[yellow]No results found for query: {query}[/yellow]")
        return
    
    rprint(f"\n[green]Search Results for:[/green] '{query}' ([cyan]{search_type}[/cyan])")
    
    for i, result in enumerate(results, 1):
        content = result.get('content', '')
        if len(content) > 200:
            content = content[:200] + "..."
        
        score_info = ""
        if result.get('score') is not None:
            score_info += f" (Score: {result['score']:.3f})"
        if result.get('combined_score') is not None:
            score_info += f" (Combined: {result['combined_score']:.3f})"
        
        panel_content = f"""
[bold]Type:[/bold] {result.get('content_type', 'unknown')}
[bold]Title:[/bold] {result.get('title', 'Unknown')}
[bold]Page:[/bold] {result.get('page', 'N/A')}
[bold]Content:[/bold] {content}
"""
        
        if result.get('latex'):
            panel_content += f"[bold]LaTeX:[/bold] {result['latex']}\n"
        
        console.print(Panel(
            panel_content, 
            title=f"Result {i}{score_info}", 
            expand=False
        ))


if __name__ == "__main__":
    app()
```


Now let me create the main entry point and some additional configuration files:

Failed creating `main.py`
- File already exists

## Me

> Context:
> - <file>mcp-build.md</file>
> - <file>mcp-build.md</file>
> - <file>build-progress.md</file>
> - <group>full_stack_dev</group>
> - <group>neovim</group>

What was the code you were goign to write in main.py?

## CodeCompanion (Copilot)

I was going to create a main.py file with this content:

```python
#!/usr/bin/env python3
"""
MCP RAG Server
==============

Main entry point for the MCP RAG Server application.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.cli import app

if __name__ == "__main__":
    app()
```

Let me continue by creating the environment configuration file and some additional setup files:

Created file `.env.example`
```example
# MCP RAG Server Environment Configuration
# Copy this file to .env and modify as needed

# Database Configuration
LANCEDB_PATH=./data/lancedb
VECTOR_DIMENSION=1024

# Model Configuration
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
MAX_BATCH_SIZE=32

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=8000
JWT_SECRET_KEY=your-secret-key-change-in-production-please-use-strong-key
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Document Processing Configuration
MAX_FILE_SIZE_MB=100

# Logging Configuration
LOG_LEVEL=INFO

# Optional: Hugging Face Cache Directory (for models)
# HF_HOME=./models/cache

# Optional: CUDA Configuration
# CUDA_VISIBLE_DEVICES=0
```


Failed creating `README.md`
- File already exists

## Me

> Context:
> - <file>mcp-build.md</file>
> - <file>mcp-build.md</file>
> - <file>build-progress.md</file>
> - <group>full_stack_dev</group>
> - <group>neovim</group>

What was the code you were going to put in the readme.md file?

## CodeCompanion (Copilot)

I was going to create a comprehensive README.md file with this content:

```markdown
# MCP RAG Server

A comprehensive document processing and retrieval system designed to assist with coding applications through intelligent document search and context augmentation.

## Features

- **Document Processing**: Extract text, structure, and equations from PDFs, text files, and Markdown
- **Flexible Embedding**: Support for multiple embedding models with easy switching
- **Vector Database**: LanceDB with efficient similarity search and hybrid retrieval
- **FastAPI Server**: RESTful API with rate limiting, authentication, and context augmentation
- **CLI Interface**: Command-line tools for document processing and server management
- **LaTeX Support**: Automatic equation detection and LaTeX formatting

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

### 2. Process a Document

```bash
# Process and embed a document in one step
python main.py process-and-embed /path/to/document.pdf

# Or process in steps
python main.py process-document /path/to/document.pdf
python main.py embed-document ./data/processed/document_processed.json
```

### 3. Start the Server

```bash
# Start the MCP server
python main.py start-server

# Or with custom settings
python main.py start-server --host 0.0.0.0 --port 8080
```

### 4. Search Documents

```bash
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

```bash
# Get access token (development only)
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "demo_user"}'
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
  "limit": 10
}

# Hybrid search (recommended)
POST /search/hybrid
{
  "query": "deep learning",
  "table_name": "documents",
  "limit": 10,
  "semantic_weight": 0.7
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

## Document Types

### Supported Formats
- **PDF**: Full text extraction with structure detection
- **Text files**: Plain text processing
- **Markdown**: Markdown with equation support

### Structure Detection
- Chapter headings (multiple patterns)
- Section numbering
- LaTeX equations (inline and display)
- Bibliography sections

### Equation Processing
Automatic detection and LaTeX formatting:
- Inline math: `$equation$`
- Display math: `$$equation$$`
- Equation environments: `\begin{equation}...\end{equation}`
- Align environments: `\begin{align}...\end{align}`

## Search Types

### Semantic Search
Uses vector embeddings for meaning-based retrieval:
```bash
python main.py search "optimization techniques" --type semantic
```

### Keyword Search
Traditional text-based search:
```bash
python main.py search "gradient descent" --type keyword
```

### Hybrid Search (Recommended)
Combines semantic and keyword approaches:
```bash
python main.py search "machine learning" --type hybrid
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

### Testing

```bash
# Run tests (when implemented)
python -m pytest tests/

# Test document processing
python main.py process-document test_document.pdf --verbose

# Test embedding
python main.py embed-document processed_doc.json --verbose

# Test search
python main.py search "test query" --verbose
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

### Performance Tips

1. **Batch Processing**: Use `batch-process` for multiple documents
2. **GPU Acceleration**: Install PyTorch with CUDA support
3. **Model Selection**: Choose appropriate embedding model for your use case
4. **Index Optimization**: LanceDB automatically optimizes vector indices

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]

## Support

[Add support information here]
```

Let me continue by creating some test files and finishing the implementation:

Created file `tests/test_mcp_rag.py`
```py
import pytest
import tempfile
import json
from pathlib import Path
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.document_processing import DocumentProcessor
from src.embedding import EmbeddingManager, create_embedder
from src.db import LanceDBHandler, create_db_handler
from src.utils import TextCleaner, EquationParser, MetadataExtractor


class TestDocumentProcessor:
    """Test document processing functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.processor = DocumentProcessor()
    
    def test_detect_chapter(self):
        """Test chapter detection."""
        assert self.processor._detect_chapter("Chapter 1: Introduction")
        assert self.processor._detect_chapter("CHAPTER 2 Data Structures")
        assert not self.processor._detect_chapter("This is just a paragraph")
    
    def test_detect_section(self):
        """Test section detection."""
        assert self.processor._detect_section("1.1 Overview")
        assert self.processor._detect_section("Section 2.3")
        assert not self.processor._detect_section("Regular text here")
    
    def test_extract_equations(self):
        """Test equation extraction."""
        text = "The formula is $E = mc^2$ and also $$\\sum_{i=1}^n x_i$$"
        equations = self.processor._extract_equations(text)
        assert len(equations) >= 1
        assert any("E = mc^2" in eq for eq in equations)
    
    def test_clean_equation(self):
        """Test equation cleaning."""
        dirty_eq = "  E = mc^2   \\\\  "
        clean_eq = self.processor._clean_equation(dirty_eq)
        assert clean_eq == "E = mc^2 \\"
    
    def test_process_text_file(self):
        """Test processing text files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Test Document\n\nThis is a test with $x = y + z$ equation.")
            f.flush()
            
            temp_path = Path(f.name)
            
        try:
            result = self.processor._process_text_file(temp_path)
            assert result['title'] == temp_path.stem
            assert len(result['equations']) >= 1
        finally:
            temp_path.unlink()


class TestEmbeddingManager:
    """Test embedding functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Use a small sentence transformer for testing
        self.embedding_manager = create_embedder(
            model_type="sentence_transformers", 
            model_name="all-MiniLM-L6-v2"
        )
    
    def test_embed_single_text(self):
        """Test single text embedding."""
        text = "This is a test sentence."
        embedding = self.embedding_manager.embed_query(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1
        assert embedding.shape[0] > 0
    
    def test_embed_multiple_texts(self):
        """Test multiple text embedding."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        embeddings = self.embedding_manager.embed_texts(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0
    
    def test_embed_document(self):
        """Test document embedding."""
        document_data = {
            'title': 'Test Document',
            'raw_text': 'This is test content.',
            'chapters': [
                {'title': 'Chapter 1', 'content': 'Chapter content'}
            ],
            'sections': [
                {'title': 'Section 1.1', 'content': 'Section content'}
            ],
            'equations': [
                {'latex': 'x = y + z', 'context': 'The equation is x = y + z'}
            ]
        }
        
        embedded_data = self.embedding_manager.embed_document(document_data)
        
        assert 'document_embedding' in embedded_data
        assert 'summary_embedding' in embedded_data
        assert embedded_data['chapters'][0].get('embedding') is not None
        assert embedded_data['sections'][0].get('embedding') is not None
        assert embedded_data['equations'][0].get('embedding') is not None


class TestLanceDBHandler:
    """Test database functionality."""
    
    def setup_method(self):
        """Setup for each test method."""
        # Use temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.db_handler = LanceDBHandler(str(self.temp_dir))
    
    def teardown_method(self):
        """Cleanup after each test."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_create_document_table(self):
        """Test table creation."""
        table = self.db_handler.create_document_table("test_table")
        assert table is not None
        assert "test_table" in self.db_handler.db.table_names()
    
    def test_insert_and_search(self):
        """Test document insertion and search."""
        # Create sample embedded document
        embedded_data = {
            'title': 'Test Document',
            'source': 'test.pdf',
            'raw_text': 'This is test content for searching.',
            'total_pages': 1,
            'chapters': [],
            'sections': [],
            'equations': [],
            'document_embedding': np.random.rand(384).tolist(),  # MiniLM dimension
            'summary': 'Test document summary',
            'summary_embedding': np.random.rand(384).tolist(),
            'metadata': {}
        }
        
        # Insert document
        doc_id = self.db_handler.insert_document(embedded_data, "test_table")
        assert doc_id is not None
        
        # Test semantic search
        query_embedding = np.random.rand(384)
        results = self.db_handler.semantic_search(query_embedding, "test_table", limit=5)
        assert len(results) >= 0  # May be empty if no similar results
    
    def test_list_documents(self):
        """Test document listing."""
        # Create table first
        self.db_handler.create_document_table("test_table")
        
        # List documents (should be empty initially)
        documents = self.db_handler.list_documents("test_table")
        assert isinstance(documents, list)


class TestUtils:
    """Test utility functions."""
    
    def test_text_cleaner(self):
        """Test text cleaning utilities."""
        dirty_text = "  This   is    messy    text...   "
        clean_text = TextCleaner.clean_text(dirty_text)
        assert clean_text == "This is messy text."
    
    def test_chunk_text(self):
        """Test text chunking."""
        long_text = "This is a very long text. " * 100
        chunks = TextCleaner.chunk_text(long_text, max_length=100, overlap=20)
        assert len(chunks) > 1
        assert all(len(chunk) <= 120 for chunk in chunks)  # Allow for overlap
    
    def test_equation_parser(self):
        """Test equation parsing."""
        text = "Here is an equation: $E = mc^2$ and another: $$\\int_0^1 x dx$$"
        equations = EquationParser.extract_equations(text)
        assert len(equations) >= 2
        assert any("E = mc^2" in eq['latex'] for eq in equations)
    
    def test_metadata_extractor(self):
        """Test metadata extraction."""
        text = "Title: Machine Learning Basics\n\nThis is a document about ML."
        title = MetadataExtractor.extract_title(text)
        assert title == "Machine Learning Basics"
        
        keywords = MetadataExtractor.extract_keywords(text, max_keywords=5)
        assert isinstance(keywords, list)
        assert len(keywords) <= 5


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def setup_method(self):
        """Setup for integration tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.processor = DocumentProcessor()
        self.embedding_manager = create_embedder(
            model_type="sentence_transformers",
            model_name="all-MiniLM-L6-v2"
        )
        self.db_handler = LanceDBHandler(str(self.temp_dir))
    
    def teardown_method(self):
        """Cleanup after integration tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_pipeline(self):
        """Test the complete document processing pipeline."""
        # Create a test document
        test_content = """
        # Machine Learning Introduction
        
        This chapter covers basic concepts of machine learning.
        
        ## 1.1 Linear Regression
        
        The equation for linear regression is $y = mx + b$ where:
        - y is the output
        - x is the input
        - m is the slope
        - b is the intercept
        
        For multiple variables: $$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\epsilon$$
        """
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(test_content)
            f.flush()
            temp_path = Path(f.name)
        
        try:
            # Step 1: Process document
            document_data = self.processor.process_document(temp_path)
            assert document_data['title'] is not None
            assert len(document_data['equations']) >= 2
            
            # Step 2: Generate embeddings
            embedded_data = self.embedding_manager.embed_document(document_data)
            assert 'document_embedding' in embedded_data
            assert 'summary_embedding' in embedded_data
            
            # Step 3: Store in database
            doc_id = self.db_handler.insert_document(embedded_data, "test_documents")
            assert doc_id is not None
            
            # Step 4: Search
            query = "linear regression equation"
            query_embedding = self.embedding_manager.embed_query(query)
            results = self.db_handler.semantic_search(
                query_embedding, "test_documents", limit=5
            )
            
            # Should find our document
            assert len(results) > 0
            found_titles = [r['title'] for r in results]
            assert any('Machine Learning' in title for title in found_titles)
            
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
```


