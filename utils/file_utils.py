import os
import shutil
import zipfile
import fitz  # PyMuPDF
import pdfplumber
import docx2txt

def save_uploaded_file(uploaded_file, save_path):
    """Persist an uploaded file object to ``save_path``.

    The function supports both Streamlit ``UploadedFile`` objects which expose
    a ``getbuffer`` method and FastAPI ``UploadFile`` objects which expose a
    ``file`` attribute. Any other file-like object with a ``read`` method is
    also supported. All data is written in binary mode.
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        if hasattr(uploaded_file, "getbuffer"):
            # Streamlit's UploadedFile
            f.write(uploaded_file.getbuffer())
        elif hasattr(uploaded_file, "file"):
            # FastAPI's UploadFile
            uploaded_file.file.seek(0)
            shutil.copyfileobj(uploaded_file.file, f)
        else:
            # Generic file-like object
            f.write(uploaded_file.read())
    return save_path

def unzip_resumes(zip_path, extract_to):
    """Unzip resumes and return a list of all extracted file paths.

    The target directory is cleared first so that results from previous
    uploads do not linger and create duplicate candidates.
    """

    if os.path.exists(extract_to):
        shutil.rmtree(extract_to)
    os.makedirs(extract_to, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile as e:
        raise ValueError("Invalid ZIP file. Please upload a valid .zip archive.") from e

    paths = []
    for root, _, files in os.walk(extract_to):
        for name in files:
            paths.append(os.path.join(root, name))
    return paths

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file.

    Raises a ``ValueError`` if the file cannot be read as a PDF.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "".join([page.extract_text() or "" for page in pdf.pages])
        return text
    except Exception as e:
        # Attempt a fallback using PyMuPDF before giving up
        try:
            text = ""
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            return text
        except Exception:
            raise ValueError("Unable to read PDF file. Ensure the file is a valid PDF.") from e

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file.

    Raises a ``ValueError`` if the file cannot be parsed.
    """
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        raise ValueError("Unable to read DOCX file. Ensure the file is a valid DOCX document.") from e
