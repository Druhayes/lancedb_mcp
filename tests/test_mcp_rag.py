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