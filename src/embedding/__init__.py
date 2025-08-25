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