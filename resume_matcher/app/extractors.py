'''
extractors.py

This module provides utility functions to extract and clean text from PDF files.
'''
import pdfplumber
import re

def extract_text_from_pdf(pdf_file):
    '''
    Extract text from a PDF file.

    Parameters:
    pdf_file (str or file-like): Path to the PDF file or a file-like object.

    Returns:
    str: A string containing the concatenated text from all pages of the PDF.
    '''
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def clean_text(text):
    '''
    Clean and normalize a block of text.

    This function converts text to lowercase, removes line breaks, bullets,
    hyphens, and extra whitespace, then strips leading/trailing spaces.

    Parameters:
    text (str): The raw text to be cleaned.

    Returns:
    str: The cleaned and normalized text.
    '''
    text = text.lower()
    text = re.sub(r"[\n•\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text