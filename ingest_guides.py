"""Chunk the PDF guide(s) at 600 chars / 100 overlap into the `guides` collection (FR-16).

Every *.pdf in data/ is ingested, so ops can add a guide by dropping in a file.
"""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

import config
from vectorstore import replace_documents

_HEADER = re.compile(r"^.*Internal Use Only\s*$", re.MULTILINE)
_PAGE_FOOTER = re.compile(r"^\s*Page \d+\s*$", re.MULTILINE)


def clean(text: str) -> str:
    text = _HEADER.sub("", text)
    text = _PAGE_FOOTER.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def load_guides() -> tuple[list[Document], list[str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    docs, ids = [], []
    for pdf in sorted(config.DATA_DIR.glob("*.pdf")):
        reader = PdfReader(pdf)
        for page_no, page in enumerate(reader.pages, start=1):
            text = clean(page.extract_text() or "")
            if len(text) < 40:          # skip cover / near-empty pages
                continue
            for i, chunk in enumerate(splitter.split_text(text)):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": "GUIDE",
                        "ref": f"GUIDE {pdf.stem} p.{page_no}",
                        "title": f"{pdf.name}, page {page_no}",
                        "page": page_no,
                    },
                ))
                ids.append(f"guide-{pdf.stem}-p{page_no}-c{i}")
    return docs, ids


def main() -> int:
    docs, ids = load_guides()
    n = replace_documents("guides", docs, ids)
    print(f"guides: {n} chunks")
    return n


if __name__ == "__main__":
    main()
