import sys

from src.umbrella_rag.ingestion.pipeline import run_ingestion


def _usage() -> None:
    print("Usage: python3 scripts/ingest.py <pdf_path>")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    run_ingestion(sys.argv[1])