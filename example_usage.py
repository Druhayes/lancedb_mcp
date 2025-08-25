#!/usr/bin/env python3
"""
MCP RAG Server - Example Usage
==============================

This script demonstrates how to use the MCP RAG server for document processing,
embedding, and retrieval in a typical workflow.
"""

import asyncio
import json
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.document_processing import DocumentProcessor
from src.embedding import create_embedder
from src.db import create_db_handler
from src.utils import Logger

# Setup logging
logger = Logger.setup_logger("example_usage")


async def example_workflow():
    """Demonstrate a complete workflow of the MCP RAG server."""
    
    logger.info("Starting MCP RAG Server example workflow...")
    
    # Step 1: Create a sample document for demonstration
    sample_doc_content = """
    # Advanced Machine Learning Concepts
    
    ## Chapter 1: Neural Networks
    
    Neural networks are computing systems inspired by biological neural networks.
    
    ### 1.1 Perceptron
    
    The basic perceptron uses the following equation:
    
    $$y = f(\\sum_{i=1}^n w_i x_i + b)$$
    
    Where:
    - $w_i$ are the weights
    - $x_i$ are the inputs  
    - $b$ is the bias term
    - $f$ is the activation function
    
    ### 1.2 Backpropagation
    
    The backpropagation algorithm computes gradients using the chain rule:
    
    $$\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial y} \\frac{\\partial y}{\\partial w}$$
    
    ## Chapter 2: Deep Learning
    
    Deep learning uses multiple layers to learn complex patterns.
    
    ### 2.1 Convolutional Neural Networks
    
    CNNs use convolution operations defined as:
    
    $$(f * g)(t) = \\int_{-\\infty}^{\\infty} f(\\tau) g(t - \\tau) d\\tau$$
    
    ### 2.2 Attention Mechanisms
    
    The attention mechanism computes weighted averages:
    
    $$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$
    
    ## Bibliography
    
    1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
    2. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436-444.
    """
    
    # Create sample document file
    sample_doc_path = Path("./data/raw/sample_ml_textbook.md")
    sample_doc_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sample_doc_path, 'w', encoding='utf-8') as f:
        f.write(sample_doc_content)
    
    logger.info(f"Created sample document: {sample_doc_path}")
    
    try:
        # Step 2: Initialize components
        logger.info("Initializing document processor...")
        processor = DocumentProcessor()
        
        logger.info("Initializing embedding manager...")
        embedding_manager = create_embedder(
            model_type="sentence_transformers",
            model_name="all-MiniLM-L6-v2"  # Smaller model for faster demo
        )
        
        logger.info("Initializing database handler...")
        # Get the actual embedding dimension from the embedding manager
        embedding_dim = embedding_manager.embedder.embedding_dim
        db_handler = create_db_handler(vector_dimension=embedding_dim)
        
        # Step 3: Process the document
        logger.info("Processing document...")
        document_data = processor.process_document(sample_doc_path)
        
        print(f"\n📄 Document Processing Results:")
        print(f"   Title: {document_data['title']}")
        print(f"   Chapters: {len(document_data['chapters'])}")
        print(f"   Sections: {len(document_data['sections'])}")
        print(f"   Equations: {len(document_data['equations'])}")
        print(f"   Bibliography entries: {len(document_data['bibliography'])}")
        
        # Show some extracted equations
        if document_data['equations']:
            print(f"\n🧮 Sample Equations:")
            for i, eq in enumerate(document_data['equations'][:3]):
                print(f"   {i+1}. {eq['latex']}")
        
        # Step 4: Generate embeddings
        logger.info("Generating embeddings...")
        embedded_data = embedding_manager.embed_document(document_data)
        
        print(f"\n🔢 Embedding Results:")
        print(f"   Document embedding dimension: {len(embedded_data['document_embedding'])}")
        print(f"   Chapter embeddings: {sum(1 for ch in embedded_data['chapters'] if ch.get('embedding'))}")
        print(f"   Section embeddings: {sum(1 for sec in embedded_data['sections'] if sec.get('embedding'))}")
        print(f"   Equation embeddings: {sum(1 for eq in embedded_data['equations'] if eq.get('embedding'))}")
        
        # Step 5: Store in database
        logger.info("Storing in database...")
        table_name = "ml_textbooks"
        doc_id = db_handler.insert_document(embedded_data, table_name)
        
        print(f"\n💾 Database Storage:")
        print(f"   Document ID: {doc_id}")
        print(f"   Table: {table_name}")
        
        # Step 6: Demonstrate different search types
        print(f"\n🔍 Search Demonstrations:")
        
        # Test queries
        test_queries = [
            "neural network equations",
            "backpropagation algorithm",
            "attention mechanism formula",
            "convolutional neural networks",
            "deep learning concepts"
        ]
        
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            
            # Semantic search
            query_embedding = embedding_manager.embed_query(query)
            semantic_results = db_handler.semantic_search(
                query_embedding, table_name, limit=3
            )
            
            # Keyword search
            keyword_results = db_handler.keyword_search(query, table_name, limit=3)
            
            # Hybrid search
            hybrid_results = db_handler.hybrid_search(
                query, query_embedding, table_name, limit=3
            )
            
            print(f"      Semantic results: {len(semantic_results)}")
            print(f"      Keyword results: {len(keyword_results)}")
            print(f"      Hybrid results: {len(hybrid_results)}")
            
            # Show best result
            if hybrid_results:
                best_result = hybrid_results[0]
                content_preview = best_result['content'][:100] + "..." if len(best_result['content']) > 100 else best_result['content']
                print(f"      Best match: {best_result['content_type']} - {content_preview}")
                if best_result.get('latex'):
                    print(f"      LaTeX: {best_result['latex']}")
        
        # Step 7: Demonstrate document listing
        print(f"\n📋 Document Management:")
        documents = db_handler.list_documents(table_name)
        print(f"   Documents in '{table_name}': {len(documents)}")
        for doc in documents:
            print(f"      - {doc['title']} (ID: {doc['id']})")
        
        # Step 8: Show equation-specific search
        print(f"\n🧮 Equation-Specific Search:")
        equation_query = "attention formula"
        query_embedding = embedding_manager.embed_query(equation_query)
        equation_results = db_handler.semantic_search(
            query_embedding, table_name, limit=5, content_type="equation"
        )
        
        print(f"   Found {len(equation_results)} equation matches for '{equation_query}':")
        for result in equation_results:
            if result.get('latex'):
                print(f"      - {result['latex']}")
                print(f"        Context: {result['content'][:80]}...")
        
        # Step 9: Demonstrate summary search
        print(f"\n📝 Summary Search:")
        summary_table = db_handler.get_or_create_summary_table()
        
        # Search summaries for documents about specific topics
        summary_query = "machine learning neural networks"
        summary_embedding = embedding_manager.embed_query(summary_query)
        summary_results = db_handler.semantic_search(
            summary_embedding, "document_summaries", limit=3
        )
        
        print(f"   Found {len(summary_results)} summary matches:")
        for result in summary_results:
            print(f"      - {result['title']}")
            print(f"        Summary: {result['content'][:100]}...")
        
        print(f"\n✅ Example workflow completed successfully!")
        print(f"\n💡 Next steps:")
        print(f"   1. Start the server: python main.py start-server")
        print(f"   2. Get an auth token: curl -X POST http://localhost:8000/auth/token")
        print(f"   3. Search via API: curl -X POST http://localhost:8000/search/hybrid \\")
        print(f"                          -H 'Authorization: Bearer <token>' \\")
        print(f"                          -H 'Content-Type: application/json' \\")
        print(f"                          -d '{{\"query\": \"neural networks\", \"table_name\": \"{table_name}\"}}'")
        
    except Exception as e:
        logger.error(f"Error in example workflow: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return False
    
    return True


def demonstrate_api_usage():
    """Show how to interact with the API programmatically."""
    
    print(f"\n🔌 API Usage Examples:")
    print(f"\n1. Get Authentication Token:")
    print(f"""
    curl -X POST "http://localhost:8000/auth/token" \\
         -H "Content-Type: application/json" \\
         -d '{{"username": "demo_user"}}'
    """)
    
    print(f"\n2. Semantic Search:")
    print(f"""
    curl -X POST "http://localhost:8000/search/semantic" \\
         -H "Authorization: Bearer <your-token>" \\
         -H "Content-Type: application/json" \\
         -d '{{
             "query": "neural network backpropagation",
             "table_name": "ml_textbooks",
             "limit": 5
         }}'
    """)
    
    print(f"\n3. Hybrid Search with Filters:")
    print(f"""
    curl -X POST "http://localhost:8000/search/hybrid" \\
         -H "Authorization: Bearer <your-token>" \\
         -H "Content-Type: application/json" \\
         -d '{{
             "query": "attention mechanism",
             "table_name": "ml_textbooks",
             "limit": 10,
             "semantic_weight": 0.8,
             "content_type": "section"
         }}'
    """)
    
    print(f"\n4. Submit Feedback:")
    print(f"""
    curl -X POST "http://localhost:8000/feedback" \\
         -H "Authorization: Bearer <your-token>" \\
         -H "Content-Type: application/json" \\
         -d '{{
             "query": "neural networks",
             "result_id": "doc_123_sec_1",
             "rating": 5,
             "comments": "Very relevant result"
         }}'
    """)
    
    print(f"\n5. List Documents:")
    print(f"""
    curl -X GET "http://localhost:8000/documents/ml_textbooks" \\
         -H "Authorization: Bearer <your-token>"
    """)


def demonstrate_cli_usage():
    """Show CLI usage examples."""
    
    print(f"\n🖥️  CLI Usage Examples:")
    print(f"\n1. Process a single document:")
    print(f"   python main.py process-document ./data/raw/textbook.pdf")
    
    print(f"\n2. Process and embed in one step:")
    print(f"   python main.py process-and-embed ./data/raw/textbook.pdf --table my_books")
    
    print(f"\n3. Batch process multiple documents:")
    print(f"   python main.py batch-process ./documents/ --pattern '*.pdf' --table textbooks")
    
    print(f"\n4. Search documents:")
    print(f"   python main.py search 'machine learning algorithms' --type hybrid --limit 5")
    
    print(f"\n5. List available documents:")
    print(f"   python main.py list-documents --table textbooks")
    
    print(f"\n6. Start the server:")
    print(f"   python main.py start-server --host 0.0.0.0 --port 8080")


if __name__ == "__main__":
    print("🚀 MCP RAG Server - Example Usage")
    print("=" * 50)
    
    # Run the example workflow
    success = asyncio.run(example_workflow())
    
    if success:
        # Show additional usage examples
        demonstrate_cli_usage()
        demonstrate_api_usage()
        
        print(f"\n📚 Additional Resources:")
        print(f"   - API Documentation: http://localhost:8000/docs")
        print(f"   - Health Check: http://localhost:8000/health")
        print(f"   - Configuration: Edit .env file")
        print(f"   - Logs: Check console output or log files")
    
    print(f"\n🎯 Happy coding with your MCP RAG Server!")
