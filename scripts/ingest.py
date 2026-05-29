import logging
from glob import glob
from pathlib import Path
import sys

from umbrella_rag.ingestion.pipeline import run_ingestion, run_ingestion_batch


def _usage() -> None:
    print("Usage: python3 scripts/ingest.py <pdf_path>")
    print("       python3 scripts/ingest.py --batch <dir_or_file> [more_paths...]")


def _collect_pdf_paths(paths: list[str]) -> list[str]:
    """
    Expand input paths into a list of PDF file paths.

    Args:
        paths: List of file or directory paths.

    Returns:
        List of PDF file paths.

    Notes:
        Directories are expanded to matching *.pdf files (non-recursive).
    """
    collected: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            collected.extend(sorted(glob(str(path / "*.pdf"))))
        elif path.is_file():
            collected.append(str(path))
        else:
            collected.append(str(path))
    return collected


def _print_summary(results: dict[str, int | str]) -> None:
    """
    Print a results summary table to stdout.

    Args:
        results: Mapping of file path to chunk count or error message.

    Returns:
        None.

    Notes:
        Uses a fixed-width column format for readability.
    """
    display_names = [Path(path).name for path in results]
    name_width = max([len("FILE")] + [len(name) for name in display_names])

    print(f"{'FILE':<{name_width}}  {'CHUNKS':<8}  STATUS")
    for path, display_name in zip(results.keys(), display_names):
        result = results[path]
        if isinstance(result, int):
            chunks = str(result)
            status = "OK"
        else:
            chunks = ""
            status = f"ERROR: {result}"
        print(f"{display_name:<{name_width}}  {chunks:<8}  {status}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)

    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            _usage()
            sys.exit(1)
        pdf_paths = _collect_pdf_paths(sys.argv[2:])
        if not pdf_paths:
            print("No PDF files found for batch ingestion.")
            sys.exit(1)
        results = run_ingestion_batch(pdf_paths)
        _print_summary(results)
        has_failure = any(isinstance(value, str) for value in results.values())
        sys.exit(1 if has_failure else 0)

    run_ingestion(sys.argv[1])