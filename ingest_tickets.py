"""Load resolved tickets from data/tickets.db: one ticket = one document (FR-15).

Only the issue and its resolution are indexed - no customer identifiers.
"""
import re
import sqlite3

from langchain_core.documents import Document

import config
from vectorstore import replace_documents

# Defensive scrub in case free-text fields ever contain personal data.
_PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\+?\d[\d\s-]{8,}\d"), "[number]"),
]


def scrub(text: str) -> str:
    for pattern, repl in _PII:
        text = pattern.sub(repl, text)
    return text


def load_tickets() -> tuple[list[Document], list[str]]:
    con = sqlite3.connect(config.TICKETS_DB)
    try:
        rows = con.execute(
            "SELECT ticket_id, category, issue_type, description, resolution "
            "FROM tickets WHERE status = 'resolved'"
        ).fetchall()
    finally:
        con.close()

    docs, ids = [], []
    for ticket_id, category, issue_type, description, resolution in rows:
        docs.append(Document(
            page_content=(
                f"Issue: {issue_type}\n"
                f"Description: {scrub(description)}\n"
                f"Resolution: {scrub(resolution)}"
            ),
            metadata={
                "source": "TICKET",
                "ref": f"TICKET {ticket_id}",
                "title": issue_type,
                "category": category,
            },
        ))
        ids.append(f"ticket-{ticket_id}")
    return docs, ids


def main() -> int:
    docs, ids = load_tickets()
    n = replace_documents("tickets", docs, ids)
    print(f"tickets: {n} documents")
    return n


if __name__ == "__main__":
    main()
