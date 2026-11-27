import io
import re

import pdfplumber
import pymupdf

REFERENCE_HEADING = re.compile(
    r"^\s*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX\d]+[.)]\s*)?"
    r"(참\s*고\s*문\s*헌|인\s*용\s*문\s*헌|references?|bibliography|works\s+cited|literature\s+cited)"
    r"\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

MIN_TEXT_CHARS = 200
# Scanned or garbled PDFs are sent to the AI as a file instead; these keep that request within API limits.
MAX_PDF_PAGES = 100
MAX_PDF_MB = 20


def extract_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Return the better of pdfplumber's and PyMuPDF's text, plus the page count.

    Raises ValueError when neither library can open the file.
    """
    candidates = []
    for extractor in (_extract_pdfplumber, _extract_pymupdf):
        try:
            candidates.append(extractor(pdf_bytes))
        except Exception:
            continue
    if not candidates:
        raise ValueError("PDF 파일을 열 수 없습니다. 파일이 손상되었거나 암호가 걸려 있을 수 있습니다.")
    text, pages = max(candidates, key=lambda c: _readable_chars(c[0]))
    return _clean(text), pages


def is_readable(text: str) -> bool:
    """False for scanned PDFs (almost no text) and PDFs whose fonts extract as garbage."""
    return _readable_chars(text) >= MIN_TEXT_CHARS and _garbage_ratio(text) < 0.1


def split_references(text: str) -> tuple[str, str]:
    """Split into (body, references). References is "" when no heading is found."""
    matches = list(REFERENCE_HEADING.finditer(text))
    if not matches:
        return text, ""
    start = matches[-1].start()
    # A heading in the first half is almost always a table-of-contents entry.
    if start < len(text) * 0.5:
        return text, ""
    return text[:start].rstrip(), text[start:].strip()


def chunk_text(text: str, max_chars: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + max_chars // 2, end)
            if cut != -1:
                end = cut
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def _extract_pdfplumber(pdf_bytes: bytes) -> tuple[str, int]:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages), len(pages)


def _extract_pymupdf(pdf_bytes: bytes) -> tuple[str, int]:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]
    return "\n".join(pages), len(pages)


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    garbage = len(re.findall(r"\(cid:\d+\)", text)) * 8 + text.count("�")
    return garbage / len(text)


def _readable_chars(text: str) -> int:
    return len(re.sub(r"\(cid:\d+\)|�|\s", "", text))
