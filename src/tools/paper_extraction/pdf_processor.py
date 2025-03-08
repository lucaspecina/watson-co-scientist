"""
PDF processing module for extracting and structuring content from scientific papers.
"""

import os
import re
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("co_scientist")

# Try to import PDF processing libraries with graceful fallbacks
# First add the user site-packages to path
import site
import sys
user_site = site.USER_SITE
if user_site not in sys.path:
    sys.path.append(user_site)

# Now try importing with module existence checks
import importlib.util

# Check for PyMuPDF
pymupdf_spec = importlib.util.find_spec("fitz")
if pymupdf_spec is not None:
    import fitz
    HAVE_PYMUPDF = True
else:
    logger.warning("PyMuPDF not found, PDF extraction capabilities will be limited")
    HAVE_PYMUPDF = False

# Check for Tesseract OCR
pytesseract_spec = importlib.util.find_spec("pytesseract")
pil_spec = importlib.util.find_spec("PIL.Image")
if pytesseract_spec is not None and pil_spec is not None:
    import pytesseract
    from PIL import Image
    HAVE_OCR = True
else:
    logger.warning("Tesseract OCR support not available, image text extraction will be disabled")
    HAVE_OCR = False

# Check for OpenCV
opencv_spec = importlib.util.find_spec("cv2")
numpy_spec = importlib.util.find_spec("numpy")
if opencv_spec is not None and numpy_spec is not None:
    import numpy as np
    import cv2
    HAVE_CV = True
else:
    logger.warning("OpenCV not available, advanced image processing will be disabled")
    HAVE_CV = False

@dataclass
class PaperSection:
    """Represents a section of a scientific paper."""
    title: str
    content: str
    level: int = 1  # Section hierarchy level (1 = top-level, 2 = subsection, etc.)
    section_number: Optional[str] = None  # e.g., "1.2.3"
    section_type: str = "generic"  # e.g., "abstract", "introduction", "methods", etc.

@dataclass
class Figure:
    """Represents a figure in a scientific paper."""
    id: str  # e.g., "Figure 1"
    caption: str
    page_num: int
    image_path: Optional[str] = None  # Path to extracted image file
    description: Optional[str] = None  # Generated description of figure

@dataclass
class Table:
    """Represents a table in a scientific paper."""
    id: str  # e.g., "Table 1"
    caption: str
    page_num: int
    content: Optional[str] = None  # Extracted table content
    rows: Optional[List[List[str]]] = None  # Structured table data
    
@dataclass
class Citation:
    """Represents a citation in a scientific paper."""
    id: str  # e.g., "[1]" or "(Smith et al., 2019)"
    text: str  # Full citation text
    authors: Optional[List[str]] = None
    title: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None

@dataclass
class Equation:
    """Represents a mathematical equation in a scientific paper."""
    id: Optional[str] = None  # e.g., "(1)"
    content: str = ""  # LaTeX or plain text representation
    page_num: int = 0

@dataclass
class ExtractedPaper:
    """Represents the extracted content from a scientific paper."""
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str] = field(default_factory=list)
    sections: List[PaperSection] = field(default_factory=list)
    figures: List[Figure] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    equations: List[Equation] = field(default_factory=list)
    pdf_path: Optional[str] = None
    full_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the extracted paper to a dictionary."""
        return asdict(self)
    
    def to_json(self, path: str) -> None:
        """Save the extracted paper as a JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedPaper':
        """Create an ExtractedPaper from a dictionary."""
        # Convert section dictionaries to PaperSection objects
        sections = [PaperSection(**section) for section in data.get('sections', [])]
        data['sections'] = sections
        
        # Convert figure dictionaries to Figure objects
        figures = [Figure(**figure) for figure in data.get('figures', [])]
        data['figures'] = figures
        
        # Convert table dictionaries to Table objects
        tables = [Table(**table) for table in data.get('tables', [])]
        data['tables'] = tables
        
        # Convert citation dictionaries to Citation objects
        citations = [Citation(**citation) for citation in data.get('citations', [])]
        data['citations'] = citations
        
        # Convert equation dictionaries to Equation objects
        equations = [Equation(**equation) for equation in data.get('equations', [])]
        data['equations'] = equations
        
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str) -> 'ExtractedPaper':
        """Load an ExtractedPaper from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class PDFProcessor:
    """
    Processes scientific paper PDFs to extract structured content.
    
    This class handles PDF parsing, content extraction, and organization
    of paper components (sections, figures, tables, citations, etc.)
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PDF processor.
        
        Args:
            config (Dict[str, Any], optional): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        
        # Set up storage directory for extracted images
        self.image_dir = self.config.get('image_dir', 'data/figures')
        os.makedirs(self.image_dir, exist_ok=True)
        
        # Image extraction quality
        self.image_quality = self.config.get('image_quality', 300)  # DPI
        
        # Configure section detection patterns
        self.section_patterns = {
            'abstract': r'^abstract$|^summary$',
            'introduction': r'^introduction$|^background$',
            'methods': r'^methods$|^materials\s+and\s+methods$|^methodology$|^experimental$',
            'results': r'^results$',
            'discussion': r'^discussion$',
            'conclusion': r'^conclusion$|^conclusions$',
            'references': r'^references$|^bibliography$|^literature\s+cited$',
            'acknowledgments': r'^acknowledgments$|^acknowledgements$',
            'appendix': r'^appendix$|^supplementary$',
        }
        
        # Configure figure and table caption patterns
        self.figure_pattern = r'(Fig(?:ure)?\.?\s*(\d+[\.\d]*))[\.\:]?\s*(.*?)(?:\.|$)'
        self.table_pattern = r'(Table\.?\s*(\d+[\.\d]*))[\.\:]?\s*(.*?)(?:\.|$)'
        
        # Configure equation detection patterns
        self.equation_pattern = r'\((\d+[\.\d]*)\)'
        
        # Configure citation patterns
        self.citation_patterns = [
            r'\[([\d,\s\-]+)\]',  # [1] or [1,2,3]
            r'\(([A-Za-z\s]+\s*et\s*al\.?[\s,]*\d{4})\)',  # (Smith et al., 2019)
            r'\(([A-Za-z\s]+\s+and\s+[A-Za-z\s]+[\s,]*\d{4})\)'  # (Smith and Jones, 2019)
        ]
        
        # Configure metadata extraction
        self.metadata_fields = {
            'doi': r'doi:\s*(10\.\d{4,}[\d\.]{1,}/[^\s]+)',
            'journal': r'journal\s*:\s*([^,\.]+)',
            'volume': r'volume\s*:?\s*(\d+)',
            'issue': r'issue\s*:?\s*(\d+)',
            'year': r'(?:19|20)\d{2}',
            'keywords': r'keywords\s*:([^\.]+)',
        }

    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> ExtractedPaper:
        """
        Process a PDF file to extract structured content.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (Optional[str], optional): Directory to save extracted files. Defaults to None.
            
        Returns:
            ExtractedPaper: Structured content extracted from the PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        if not HAVE_PYMUPDF:
            logger.error("PyMuPDF is required for PDF processing")
            raise ImportError("PyMuPDF is required for PDF processing")
            
        # Set output directory for extracted files
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = self.image_dir
            
        # Extract paper components
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Initialize paper extraction with basic structure
        paper = ExtractedPaper(
            title="", 
            authors=[], 
            abstract="",
            pdf_path=pdf_path
        )
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Extract paper metadata and basic structure
            paper = self._extract_basic_metadata(doc, paper)
            
            # Extract full text
            paper.full_text = self._extract_full_text(doc)
            
            # Extract sections
            paper.sections = self._extract_sections(doc)
            
            # Extract figures
            paper.figures = self._extract_figures(doc, output_dir)
            
            # Extract tables
            paper.tables = self._extract_tables(doc)
            
            # Extract citations
            paper.citations = self._extract_citations(doc)
            
            # Extract equations
            paper.equations = self._extract_equations(doc)
            
            # Extract abstract if not already found
            if not paper.abstract and paper.sections:
                # Find abstract section
                for section in paper.sections:
                    if section.section_type == 'abstract':
                        paper.abstract = section.content
                        break
            
            # Default title if not found
            if not paper.title:
                paper.title = os.path.basename(pdf_path).replace('.pdf', '')
            
            # Close the document
            doc.close()
            
            logger.info(f"Successfully processed PDF: {pdf_path}")
            return paper
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            # Return partial results if available
            return paper
            
    def _extract_basic_metadata(self, doc: 'fitz.Document', paper: ExtractedPaper) -> ExtractedPaper:
        """Extract basic metadata from the PDF."""
        try:
            # Extract metadata from PDF document
            metadata = doc.metadata
            
            # Title
            if metadata.get('title'):
                paper.title = metadata.get('title')
            
            # Authors
            if metadata.get('author'):
                # Split author string into individual authors
                authors_str = metadata.get('author', '')
                authors = [a.strip() for a in re.split(r'[;,]', authors_str) if a.strip()]
                paper.authors = authors
            
            # Extract metadata using patterns from first few pages
            metadata_text = ""
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                metadata_text += page.get_text()
            
            # Extract additional metadata using patterns
            for field, pattern in self.metadata_fields.items():
                matches = re.search(pattern, metadata_text, re.IGNORECASE)
                if matches:
                    paper.metadata[field] = matches.group(1).strip()
            
            # Handle special case for keywords
            if 'keywords' in paper.metadata:
                keywords_str = paper.metadata['keywords']
                paper.keywords = [k.strip() for k in re.split(r'[;,]', keywords_str) if k.strip()]
            
            return paper
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return paper
            
    def _extract_full_text(self, doc: 'fitz.Document') -> str:
        """Extract full text from the PDF."""
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        return full_text
            
    def _extract_sections(self, doc: 'fitz.Document') -> List[PaperSection]:
        """Extract sections from the PDF."""
        sections = []
        
        # Process each page to detect section headers and content
        current_section = None
        section_content = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]
                        is_bold = span["flags"] & 2 > 0  # Check if text is bold
                        
                        # Detect if this looks like a section header
                        # Criteria: Bold text, larger font size, short length, not ending with punctuation
                        is_header = (is_bold or font_size > 11) and len(text) < 100 and not text.endswith('.')
                        
                        if is_header:
                            # Check if text matches known section types
                            section_type = "generic"
                            for s_type, pattern in self.section_patterns.items():
                                if re.match(pattern, text.lower(), re.IGNORECASE):
                                    section_type = s_type
                                    break
                            
                            # If we've been processing a section, save it
                            if current_section and section_content:
                                sections.append(PaperSection(
                                    title=current_section,
                                    content=section_content.strip(),
                                    section_type=current_section_type
                                ))
                            
                            # Start a new section
                            current_section = text
                            current_section_type = section_type
                            section_content = ""
                        else:
                            # Add text to current section content
                            section_content += text + " "
        
        # Add the last section
        if current_section and section_content:
            sections.append(PaperSection(
                title=current_section,
                content=section_content.strip(),
                section_type=current_section_type
            ))
        
        # Handle case where no sections were detected
        if not sections and doc:
            # Default to treating the entire document as one section
            sections.append(PaperSection(
                title="Main Content",
                content=self._extract_full_text(doc),
                section_type="generic"
            ))
        
        return sections
    
    def _extract_figures(self, doc: 'fitz.Document', output_dir: str) -> List[Figure]:
        """Extract figures from the PDF."""
        figures = []
        
        # Process each page to detect images and captions
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get page text
            page_text = page.get_text()
            
            # Find figure captions using regex
            caption_matches = re.finditer(self.figure_pattern, page_text, re.IGNORECASE | re.MULTILINE)
            
            for match in caption_matches:
                figure_id = match.group(1)
                figure_num = match.group(2)
                caption = match.group(3).strip()
                
                # Extract images from the page
                image_list = page.get_images(full=True)
                
                # If images are found, save them
                if image_list:
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Save image to file
                        image_filename = f"figure_{page_num+1}_{figure_num}_{img_index}.png"
                        image_path = os.path.join(output_dir, image_filename)
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        # Add figure to list
                        figures.append(Figure(
                            id=figure_id,
                            caption=caption,
                            page_num=page_num,
                            image_path=image_path
                        ))
                        
                        # Just use the first image for this figure
                        break
                else:
                    # If no images found, still record the figure caption
                    figures.append(Figure(
                        id=figure_id,
                        caption=caption,
                        page_num=page_num
                    ))
        
        return figures
    
    def _extract_tables(self, doc: 'fitz.Document') -> List[Table]:
        """Extract tables from the PDF."""
        tables = []
        
        # Process each page to detect table captions and content
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get page text
            page_text = page.get_text()
            
            # Find table captions using regex
            caption_matches = re.finditer(self.table_pattern, page_text, re.IGNORECASE | re.MULTILINE)
            
            for match in caption_matches:
                table_id = match.group(1)
                table_num = match.group(2)
                caption = match.group(3).strip()
                
                # For now, just record the table caption
                # Table content extraction is more complex and would require additional techniques
                tables.append(Table(
                    id=table_id,
                    caption=caption,
                    page_num=page_num
                ))
        
        return tables
    
    def _extract_citations(self, doc: 'fitz.Document') -> List[Citation]:
        """Extract citations from the PDF."""
        citations = []
        citation_texts = set()  # To avoid duplicates
        
        # Process full text to detect citations
        full_text = self._extract_full_text(doc)
        
        # Try to find references section
        references_match = re.search(r'references(.*?)(?=\n\s*\n|$)', full_text, re.IGNORECASE | re.DOTALL)
        references_text = references_match.group(1) if references_match else ""
        
        # Process references section if found
        if references_text:
            # Try to extract numbered references
            num_ref_matches = re.finditer(r'(?:^|\n)(\d+)\.\s+(.+?)(?=(?:\n\d+\.)|$)', references_text, re.DOTALL)
            for match in num_ref_matches:
                ref_id = match.group(1)
                ref_text = match.group(2).strip()
                
                if ref_text and ref_text not in citation_texts:
                    citation_texts.add(ref_text)
                    citations.append(Citation(
                        id=f"[{ref_id}]",
                        text=ref_text
                    ))
        
        # If no references found in references section, try to find in-text citations
        if not citations:
            for pattern in self.citation_patterns:
                citation_matches = re.finditer(pattern, full_text)
                for match in citation_matches:
                    citation_id = match.group(1)
                    
                    # Use a 15 character window to represent the citation context
                    start_idx = max(0, match.start() - 15)
                    end_idx = min(len(full_text), match.end() + 15)
                    context = full_text[start_idx:end_idx]
                    
                    if citation_id and context not in citation_texts:
                        citation_texts.add(context)
                        citations.append(Citation(
                            id=citation_id,
                            text=context
                        ))
        
        return citations
    
    def _extract_equations(self, doc: 'fitz.Document') -> List[Equation]:
        """Extract equations from the PDF."""
        equations = []
        
        # Process each page to detect equations
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Get page text
            page_text = page.get_text()
            
            # Look for equation numbers
            eq_matches = re.finditer(self.equation_pattern, page_text)
            
            for match in eq_matches:
                eq_id = match.group(1)
                
                # Use a 50 character window to represent the equation context
                start_idx = max(0, match.start() - 50)
                end_idx = min(len(page_text), match.end() + 50)
                eq_content = page_text[start_idx:end_idx]
                
                equations.append(Equation(
                    id=f"({eq_id})",
                    content=eq_content,
                    page_num=page_num
                ))
        
        return equations
    
    def save_extracted_paper(self, paper: ExtractedPaper, output_path: str) -> None:
        """Save the extracted paper to a JSON file."""
        paper.to_json(output_path)
        logger.info(f"Saved extracted paper to {output_path}")
    
    @staticmethod
    def load_extracted_paper(input_path: str) -> ExtractedPaper:
        """Load an extracted paper from a JSON file."""
        return ExtractedPaper.from_json(input_path)