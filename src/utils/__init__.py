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