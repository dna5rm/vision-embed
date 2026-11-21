"""SQLite — exactly one file, WAL, sqlite-vec, explicit migrations (SPEC section 7).

No ORM, no Alembic, no second engine. Vectors live in sqlite-vec vec0
float[512] / float[768] columns as raw float32 blobs. Distances are computed
on read with 1 - dot over stored unit vectors (exact-match nearest scan at
this scale, section 2); L2 KNN ordering on unit vectors equals cosine ordering,
so vec0 MATCH is used for row selection and numpy for the reported distances.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

from .errors import ApiError
from .vectors import blob_to_vec, to_float32_unit, vec_to_blob

MIGRATIONS: list[tuple[str, str]] = [
    (
        "001_create_tables",
        """
        CREATE TABLE assets (
          id INTEGER PRIMARY KEY,
          path TEXT UNIQUE,
          sha256 TEXT UNIQUE NOT NULL,
          width INTEGER NOT NULL,
          height INTEGER NOT NULL,
          bytes INTEGER NOT NULL,
          indexed_at TEXT NOT NULL
        );
        CREATE TABLE faces (
          id INTEGER PRIMARY KEY,
          asset_id INTEGER NOT NULL REFERENCES assets(id),
          x INTEGER NOT NULL, y INTEGER NOT NULL, w INTEGER NOT NULL, h INTEGER NOT NULL,
          det_score REAL NOT NULL,
          cluster_id INTEGER REFERENCES clusters(id)
        );
        CREATE VIRTUAL TABLE face_vec USING vec0(
          face_id INTEGER PRIMARY KEY,
          embedding float[512]
        );
        CREATE TABLE clusters (
          id INTEGER PRIMARY KEY,
          kind TEXT NOT NULL CHECK (kind IN ('face','object')),
          name TEXT,
          named_at TEXT,
          exemplar_face_id INTEGER,
          merged_into INTEGER REFERENCES clusters(id)
        );
        CREATE UNIQUE INDEX clusters_kind_name ON clusters(kind, name) WHERE name IS NOT NULL;
        CREATE TABLE objects (
          id INTEGER PRIMARY KEY,
          asset_id INTEGER NOT NULL REFERENCES assets(id),
          label TEXT NOT NULL,
          score REAL NOT NULL,
          x INTEGER NOT NULL, y INTEGER NOT NULL, w INTEGER NOT NULL, h INTEGER NOT NULL,
          cluster_id INTEGER
        );
        CREATE VIRTUAL TABLE object_vec USING vec0(
          object_id INTEGER PRIMARY KEY,
          embedding float[768]
        );
        CREATE VIRTUAL TABLE scene_vec USING vec0(
          asset_id INTEGER PRIMARY KEY,
          embedding float[768]
        );
        """,
    ),
]


class IndexStore:
    """The one SQLite connection of the one deployment (INV-1, INV-5)."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        path = Path(db_path)
        if path.parent and not path.parent.exists():
            raise ApiError("db.error", "db parent directory does not exist")
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)  # load sqlite-vec extension here (section 7)
            self.conn.enable_load_extension(False)
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("BEGIN")
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS _index_migrations ("
                "version TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (datetime('now')))"
            )
            self.conn.commit()
        except (sqlite3.Error, OSError) as exc:
            raise ApiError("db.error", f"sqlite open failed: {exc.__class__.__name__}") from None
        self.run_migrations()

    def run_migrations(self) -> None:
        applied = {row[0] for row in self.conn.execute("SELECT version FROM _index_migrations")}
        for version, ddl in MIGRATIONS:
            if version in applied:
                continue
            try:
                self.conn.execute("BEGIN")
                self.conn.executescript(ddl)
                self.conn.execute(
                    "INSERT INTO _index_migrations (version) VALUES (?)", (version,)
                )
                self.conn.commit()
            except sqlite3.Error as exc:
                self.conn.rollback()
                raise ApiError(
                    "db.error", f"migration {version} failed: {exc.__class__.__name__}"
                ) from None
        self.conn.execute("PRAGMA foreign_keys = ON")

    # -- vector storage ---------------------------------------------------

    def put_vector(self, table: str, key_col: str, key: int, dim: int, values) -> None:
        vec = to_float32_unit(values)
        if vec.shape[0] != dim:
            raise ApiError("db.error", "vector dimension mismatch")
        self.conn.execute(
            f"INSERT OR REPLACE INTO {table}({key_col}, embedding) VALUES (?, ?)",
            (key, vec_to_blob(vec)),
        )

    def get_vectors(self, table: str, key_col: str, keys=None) -> dict[int, np.ndarray]:
        if keys is None:
            rows = self.conn.execute(f"SELECT {key_col}, embedding FROM {table}").fetchall()
        else:
            rows = self.conn.execute(
                f"SELECT {key_col}, embedding FROM {table} WHERE {key_col} IN "
                f"({','.join('?' * len(keys))})",
                tuple(keys),
            ).fetchall()
        return {int(k): blob_to_vec(bytes(v)) for k, v in rows}

    # -- read-only checks for /readyz and tests ----------------------------

    def table_counts(self) -> dict[str, int]:
        out = {}
        for table in ("assets", "faces", "clusters", "objects"):
            out[table] = int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        return out

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass
