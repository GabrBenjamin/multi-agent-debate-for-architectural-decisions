"""Create the compact ADR-tracking database used by the RAG workflow."""

import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SOURCE_DB = BASE_DIR / "adr_data.db"
TARGET_DB = BASE_DIR / "adr_data_rag_minimal.db"


def main() -> None:
    if not SOURCE_DB.exists():
        raise FileNotFoundError(f"Source database not found: {SOURCE_DB}")
    if TARGET_DB.exists():
        raise FileExistsError(f"Target database already exists: {TARGET_DB}")

    with sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True) as source:
        rows = source.execute(
            """
            SELECT adr_id, adr_name, hash, points_to_adr_id, is_oldest
            FROM adr_tracking_info
            """
        )

        with sqlite3.connect(TARGET_DB) as target:
            target.execute(
                """
                CREATE TABLE adr_tracking_info (
                    adr_id TEXT NOT NULL,
                    adr_name TEXT NOT NULL,
                    hash TEXT NOT NULL,
                    points_to_adr_id TEXT,
                    is_oldest TEXT
                )
                """
            )
            target.executemany(
                "INSERT INTO adr_tracking_info VALUES (?, ?, ?, ?, ?)", rows,
            )

    print(f"Created {TARGET_DB.name}")


if __name__ == "__main__":
    main()
