"""Load data/faq.csv into the `faq` collection: one row = one document (FR-14)."""
import csv

from langchain_core.documents import Document

import config
from vectorstore import replace_documents


def load_faq() -> tuple[list[Document], list[str]]:
    docs, ids = [], []
    with open(config.FAQ_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            question, answer = row["question"].strip(), row["answer"].strip()
            if not question or not answer:
                continue
            docs.append(Document(
                page_content=f"Q: {question}\nA: {answer}",
                metadata={
                    "source": "FAQ",
                    "ref": f"FAQ #{row['id']}",
                    "title": question,
                    "category": row.get("category", ""),
                },
            ))
            ids.append(f"faq-{row['id']}")
    return docs, ids


def main() -> int:
    docs, ids = load_faq()
    n = replace_documents("faq", docs, ids)
    print(f"faq: {n} documents")
    return n


if __name__ == "__main__":
    main()
