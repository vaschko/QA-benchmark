"""Document loaders: txt / md / pdf / docx -> plain text."""

from __future__ import annotations

from pathlib import Path

SUPPORTED = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def load_document(path: str | Path) -> str:
    """Reads a document and returns its text content."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    raise ValueError(
        f"Unsupported format '{suffix}'. Supported: {sorted(SUPPORTED)}"
    )


def _load_pdf(path: Path) -> str:
    # PyMuPDF (fitz) preserves character spacing far better than pypdf, which on
    # some PDFs splits words with stray spaces ("W hy this inform ation").
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyMuPDF is missing -- please run `pip install -r requirements.txt`.") from exc

    with fitz.open(str(path)) as doc:
        pages = [page.get_text() for page in doc]
    return "\n\n".join(pages).strip()


def _load_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("python-docx is missing -- please run `pip install -r requirements.txt`.") from exc

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs).strip()
