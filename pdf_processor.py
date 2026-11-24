import io
import re

import pdfplumber

REFERENCE_HEADING = re.compile(
    r"^\s*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩIVX\d]+[.)]\s*)?"
    r"(참\s*고\s*문\s*헌|인\s*용\s*문\s*헌|references?|bibliography|works\s+cited|literature\s+cited)"
    r"\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_text(pdf_bytes: bytes) -> tuple[str, int]:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    text = "\n".join(pages).replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), len(pages)


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
