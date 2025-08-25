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