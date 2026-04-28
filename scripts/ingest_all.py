"""
Run graph and vector ingestion in sequence.

Usage (inside Docker):
    docker compose run --rm api python scripts/ingest_all.py

Usage (local, from fab-sop-rag/):
    python scripts/ingest_all.py

Exit code reflects the worst failure (0 = all OK, non-zero = at least one script failed).
"""

import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_STEPS = [
    ("Graph seed  (Neo4j)", _SCRIPTS_DIR / "ingest_graph.py"),
    ("Vector seed (Chroma)", _SCRIPTS_DIR / "ingest_vector.py"),
]


def main() -> None:
    overall_rc = 0
    for label, script in _STEPS:
        print(f"\n{'=' * 60}")
        print(f"  {label}")
        print(f"  {script}")
        print("=" * 60)
        result = subprocess.run([sys.executable, str(script)], check=False)
        if result.returncode != 0:
            print(f"\n[ERROR] {label} failed with exit code {result.returncode}")
            overall_rc = result.returncode
        else:
            print(f"\n[OK] {label} completed successfully.")

    print(f"\n{'=' * 60}")
    if overall_rc == 0:
        print("  All ingestion steps completed successfully.")
    else:
        print("  One or more ingestion steps FAILED — check output above.")
    print("=" * 60)
    sys.exit(overall_rc)


if __name__ == "__main__":
    main()
