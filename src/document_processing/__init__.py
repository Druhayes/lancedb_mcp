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