"""Rebuild every collection. Safe to re-run (FR-17)."""
import ingest_faq
import ingest_guides
import ingest_tickets

if __name__ == "__main__":
    ingest_faq.main()
    ingest_tickets.main()
    ingest_guides.main()
    print("Done. Restart the app to pick up changes.")
