"""Index logic — clustering, naming, merging, queries (SPEC section 8).

All ingest writes happen inside the single transaction of section 6b step 8;
/v1/name and /v1/merge each open their own single transaction; /v1/identify
writes nothing. No SQLite write ever occurs outside those transactions (FI40).

Face clusters only. Object instances are never auto-clustered (§8 step 6):
nothing in this module touches objects.cluster_id except name_object_instance.
"""

from __future__ import annotations

import sqlite3

import datetime as _dt

import numpy as np

from .db import IndexStore
from .errors import ApiError
from .vectors import blob_to_vec


def _now_rfc3339() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_cluster(conn, cluster_id: int) -> int | None:
    """Follow merged_into pointers to the surviving cluster (FI26)."""
    seen = set()
    current = int(cluster_id)
    while True:
        if current in seen:
            return None
        seen.add(current)
        row = conn.execute(
            "SELECT merged_into FROM clusters WHERE id = ?", (current,)
        ).fetchone()
        if row is None:
            return None
        if row[0] is None:
            return current
        current = int(row[0])


# -- ingest ------------------------------------------------------------------


def dedupe_sha(store: IndexStore, sha: str) -> dict | None:
    """Section 6b step 6: assets.sha256 lookup BEFORE the embed call."""
    row = store.conn.execute(
        "SELECT id, path, sha256, width, height, bytes, indexed_at FROM assets WHERE sha256 = ?",
        (sha,),
    ).fetchone()
    if row is None:
        return None
    asset_id, path, sha_out, width, height, nbytes, indexed_at = row
    face_cluster_ids = [
        int(r[0])
        for r in store.conn.execute(
            "SELECT COALESCE(faces.cluster_id, -1) FROM faces WHERE asset_id = ?", (asset_id,)
        ).fetchall()
    ]
    return {
        "asset_id": int(asset_id),
        "path": path,
        "sha256": sha_out,
        "width": int(width),
        "height": int(height),
        "bytes": int(nbytes),
        "indexed_at": indexed_at,
        "face_count": len(face_cluster_ids),
        "face_cluster_ids": [
            int(c) for c in face_cluster_ids if c >= 0
        ],
        "object_count": int(
            store.conn.execute(
                "SELECT COUNT(*) FROM objects WHERE asset_id = ?", (asset_id,)
            ).fetchone()[0]
        ),
    }


def match_face_vector(
    store: IndexStore, vector: np.ndarray, face_match_max: float
) -> tuple[int | None, float, bool]:
    """§8 steps 2-3 join rule on an EXISTING db: nearest non-merged face-cluster
    exemplar within face_match_max. Used by /v1/identify (writes nothing)."""
    best_id, best_dist = _nearest_exemplar(store.conn, vector)
    if best_id is not None and best_dist <= face_match_max:
        return best_id, best_dist, True
    return best_id, best_dist, False


def _nearest_exemplar(conn, vector: np.ndarray) -> tuple[int | None, float]:
    rows = conn.execute(
        """
        SELECT c.id, fv.embedding FROM clusters c
        JOIN faces f ON f.id = c.exemplar_face_id
        JOIN face_vec fv ON fv.face_id = f.id
        WHERE c.kind = 'face' AND c.merged_into IS NULL
        """
    ).fetchall()
    if not rows:
        return None, 1.0
    cands = np.vstack([blob_to_vec(bytes(r[1])) for r in rows])
    dists = 1.0 - (cands @ vector)
    idx = int(np.argmin(dists))
    return int(rows[idx][0]), float(dists[idx])


def ingest_asset(
    store: IndexStore,
    *,
    raw_bytes: int,
    sha: str,
    width: int,
    height: int,
    path: str | None,
    scene_vector,
    face_records: list[dict],
    object_records: list[dict],
    face_match_max: float,
) -> dict:
    """Section 6b step 8 — ALL ingest writes inside this one transaction,
    committed or rolled back atomically. On failure raises db.error; nothing
    partial is ever visible."""
    conn = store.conn
    try:
        conn.execute("BEGIN")
        now = _now_rfc3339()
        cursor = conn.execute(
            "INSERT INTO assets (path, sha256, width, height, bytes, indexed_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (path, sha, int(width), int(height), int(raw_bytes), now),
        )
        asset_id = int(cursor.lastrowid)
        store.put_vector("scene_vec", "asset_id", asset_id, 768, scene_vector)

        cluster_ids: list[int] = []
        clusters_created = 0
        for face in face_records:
            f = conn.execute(
                "INSERT INTO faces (asset_id, x, y, w, h, det_score, cluster_id)"
                " VALUES (?, ?, ?, ?, ?, ?, NULL)",
                (asset_id, face["bbox"]["x"], face["bbox"]["y"],
                 face["bbox"]["w"], face["bbox"]["h"], face["det_score"]),
            )
            face_id = int(f.lastrowid)
            store.put_vector("face_vec", "face_id", face_id, 512, face["vector"])
            vec = store.get_vectors("face_vec", "face_id", [face_id])[face_id]
            cluster_id, dist, created = _join_or_create(conn, store, face_id, vec, face_match_max)
            cluster_ids.append(cluster_id)
            if not created:
                clusters_created += 1

        for obj in object_records:
            obj_cur = conn.execute(
                "INSERT INTO objects (asset_id, label, score, x, y, w, h, cluster_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (asset_id, obj["label"], obj["score"],
                 obj["bbox"]["x"], obj["bbox"]["y"], obj["bbox"]["w"], obj["bbox"]["h"]),
            )
            object_id = int(obj_cur.lastrowid)
            if obj["vector"] is not None:
                store.put_vector(
                    "object_vec", "object_id", object_id, 768, obj["vector"],
                )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise ApiError("db.error", f"ingest transaction rolled back: {exc.__class__.__name__}") from None
    return {
        "asset_id": asset_id,
        "created": True,
        "face_count": len(face_records),
        "face_cluster_ids": cluster_ids,
        "object_count": len(object_records),
        "clusters_created": clusters_created,
    }


def _join_or_create(conn, store: IndexStore, face_id: int, vec: np.ndarray,
                    face_match_max: float) -> tuple[int, float, bool]:
    best_id, best_dist = _nearest_exemplar(conn, vec)
    if best_id is not None and best_dist <= face_match_max:
        conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (best_id, face_id))
        return best_id, best_dist, True
    cur = conn.execute(
        "INSERT INTO clusters (kind, name, named_at, exemplar_face_id, merged_into)"
        " VALUES ('face', NULL, NULL, ?, NULL)",
        (face_id,),
    )
    cluster_id = int(cur.lastrowid)
    conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cluster_id, face_id))
    return cluster_id, best_dist, False


# -- naming -------------------------------------------------------------------


def _validate_name(cfg_name_max: int, cfg_name_min: int, name: str) -> None:
    if len(name) > cfg_name_max:
        raise ApiError("name.too_long", "name exceeds name.max_length")
    if len(name) < cfg_name_min:
        raise ApiError("decode.failed", "name below name.min_length")


def set_cluster_name(conn, cluster_id: int, name: str, *, min_len: int, max_len: int) -> dict:
    _validate_name(max_len, min_len, name)
    row = conn.execute(
        "SELECT kind, merged_into, name, named_at FROM clusters WHERE id = ?", (cluster_id,)
    ).fetchone()
    if row is None or row[1] is not None:
        raise ApiError("not_found", "cluster_id resolves to nothing")
    kind, _merged_into, previous, named_at = row[0], row[1], row[2], row[3]
    clash = conn.execute(
        "SELECT id FROM clusters WHERE kind = ? AND name = ? AND id != ? AND merged_into IS NULL",
        (kind, name, cluster_id),
    ).fetchone()
    if clash is not None:
        raise ApiError("name.conflict", f"name already used by a {kind} cluster")
    try:
        conn.execute(
            "UPDATE clusters SET name = ?, named_at = COALESCE(named_at, ?) WHERE id = ?",
            (name, _now_rfc3339(), cluster_id),
        )
    except sqlite3.IntegrityError:
        raise ApiError("name.conflict", "name already used by a cluster of the same kind") from None
    final = conn.execute("SELECT named_at FROM clusters WHERE id = ?", (cluster_id,)).fetchone()
    return {
        "cluster_id": cluster_id,
        "kind": kind,
        "name": name,
        "previous_name": previous,
        "named_at": final[0],
    }


def create_object_cluster_for(conn, object_id: int) -> int:
    """§8 step 8 object form: create an object-kind cluster for a null-cluster
    object instance, pin exemplar_face_id = the object id by convention (it is
    not a face; the column holds the originating object id for object kind)."""
    cur = conn.execute(
        "INSERT INTO clusters (kind, name, named_at, exemplar_face_id, merged_into)"
        " VALUES ('object', NULL, NULL, ?, NULL)",
        (object_id,),
    )
    cid = int(cur.lastrowid)
    conn.execute("UPDATE objects SET cluster_id = ? WHERE id = ?", (cid, object_id))
    return cid


# -- merge --------------------------------------------------------------------


def merge_clusters(conn, from_id: int, into_id: int) -> dict:
    """§8 step 7 — face clusters only; both must exist, be unmerged. On merge
    the surviving cluster keeps its own name and exemplar untouched (FI19)."""
    if from_id == into_id:
        raise ApiError("not_found", "merge requires two distinct clusters")
    frow = conn.execute(
        "SELECT kind, merged_into, name FROM clusters WHERE id = ?", (from_id,)
    ).fetchone()
    irow = conn.execute(
        "SELECT kind, merged_into, name FROM clusters WHERE id = ?", (into_id,)
    ).fetchone()
    if frow is None or irow is None or frow[1] is not None or irow[1] is not None:
        raise ApiError("not_found", "cluster_id resolves to nothing (or already merged)")
    if frow[0] != "face":
        raise ApiError("cluster.not_face", "merge supports face clusters only")
    if irow[0] != "face":
        raise ApiError("cluster.not_face", "merge supports face clusters only")
    from_name, into_name = frow[2], irow[2]
    if from_name is not None and into_name is not None and from_name != into_name:
        raise ApiError("name.conflict", "both clusters are named and the names differ")
    moved = int(conn.execute(
        "SELECT COUNT(*) FROM faces WHERE cluster_id = ?", (from_id,)
    ).fetchone()[0])
    conn.execute("UPDATE faces SET cluster_id = ? WHERE cluster_id = ?", (into_id, from_id))
    conn.execute("UPDATE clusters SET merged_into = ? WHERE id = ?", (into_id, from_id))
    # §8: from's name always goes away — when both were named they were equal
    # (kept unique by index otherwise), and when only from was named its name
    # is deleted and the surviving cluster id stays into_id.
    conn.execute("UPDATE clusters SET name = NULL, named_at = NULL WHERE id = ?", (from_id,))
    conn.commit()
    return {
        "surviving_cluster_id": into_id,
        "merged_cluster_id": from_id,
        "face_count_moved": moved,
        "name": into_name,
    }


# -- queries ------------------------------------------------------------------


def get_asset_record(store: IndexStore, asset_id: int) -> dict:
    conn = store.conn
    row = conn.execute(
        "SELECT id, path, sha256, width, height, bytes, indexed_at FROM assets WHERE id = ?",
        (asset_id,),
    ).fetchone()
    if row is None:
        raise ApiError("not_found", f"asset {asset_id} does not exist")
    faces = conn.execute(
        "SELECT f.id, f.asset_id, f.x, f.y, f.w, f.h, f.det_score, f.cluster_id "
        "FROM faces f WHERE f.asset_id = ? ORDER BY f.id ASC",
        (asset_id,),
    ).fetchall()
    out_faces = []
    for f in faces:
        cluster_id = int(f[7]) if f[7] is not None else None
        cluster_name = None
        if cluster_id is not None:
            eff = resolve_cluster(conn, cluster_id)
            cluster_id_out = eff if eff is not None else cluster_id
            crow = conn.execute(
                "SELECT name FROM clusters WHERE id = ?", (cluster_id_out,)
            ).fetchone()
            cluster_name = crow[0] if crow is not None else None
            cluster_id = cluster_id_out
        out_faces.append({
            "id": int(f[0]), "asset_id": int(f[1]),
            "bbox": {"x": int(f[2]), "y": int(f[3]), "w": int(f[4]), "h": int(f[5])},
            "det_score": float(f[6]),
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
        })
    objects = conn.execute(
        "SELECT id, asset_id, label, score, x, y, w, h, cluster_id FROM objects"
        " WHERE asset_id = ? ORDER BY id ASC",
        (asset_id,),
    ).fetchall()
    out_objects = []
    for o in objects:
        out_objects.append({
            "id": int(o[0]), "label": o[2], "score": float(o[3]),
            "bbox": {"x": int(o[4]), "y": int(o[5]), "w": int(o[6]), "h": int(o[7])},
            "cluster_id": int(o[8]) if o[8] is not None else None,
        })
    return {
        "id": int(row[0]), "path": row[1], "sha256": row[2],
        "width": int(row[3]), "height": int(row[4]), "bytes": int(row[5]),
        "indexed_at": row[6],
        "faces": out_faces,
        "objects": out_objects,
    }


def clusters_listing(conn, kind: str, unnamed: bool | None) -> list[dict]:
    if kind not in ("face", "object"):
        raise ApiError("decode.failed", "kind must be 'face' or 'object'")
    sql = "SELECT id, kind, name FROM clusters WHERE kind = ? AND merged_into IS NULL"
    params: list = [kind]
    if unnamed is True:
        sql += " AND name IS NULL"
    elif unnamed is False:
        sql += " AND name IS NOT NULL"
    rows = conn.execute(sql, tuple(params)).fetchall()
    out = []
    for r in rows:
        cid = int(r[0])
        if r[1] == "face":
            face_count = int(conn.execute(
                "SELECT COUNT(*) FROM faces WHERE cluster_id = ?", (cid,)
            ).fetchone()[0])
            asset_count = int(conn.execute(
                "SELECT COUNT(DISTINCT asset_id) FROM faces WHERE cluster_id = ?", (cid,)
            ).fetchone()[0])
        else:
            face_count = 0
            asset_count = int(conn.execute(
                "SELECT COUNT(DISTINCT asset_id) FROM objects WHERE cluster_id = ?", (cid,)
            ).fetchone()[0])
        out.append({"id": cid, "kind": r[1], "name": r[2],
                    "face_count": face_count, "asset_count": asset_count})
    return out


def search_person(conn, name: str) -> dict | None:
    rows = conn.execute(
        "SELECT id FROM clusters WHERE kind = 'face' AND name = ? AND merged_into IS NULL",
        (name,),
    ).fetchall()
    if not rows:
        return None
    cluster_ids = [int(r[0]) for r in rows]
    faces = []
    asset_ids: list[int] = []
    for cid in cluster_ids:
        frows = conn.execute(
            "SELECT id, asset_id, cluster_id, x, y, w, h FROM faces"
            " WHERE cluster_id = ? ORDER BY id ASC",
            (cid,),
        ).fetchall()
        for f in frows:
            faces.append({
                "face_id": int(f[0]), "asset_id": int(f[1]), "cluster_id": int(f[2]),
                "bbox": {"x": int(f[3]), "y": int(f[4]), "w": int(f[5]), "h": int(f[6])},
            })
            if int(f[1]) not in asset_ids:
                asset_ids.append(int(f[1]))
    return {"clusters": cluster_ids, "faces": faces, "asset_ids": asset_ids}


def search_objects(conn, label: str) -> list[dict]:
    rows = conn.execute(
        "SELECT id, asset_id, score, x, y, w, h, cluster_id FROM objects"
        " WHERE label = ? ORDER BY id ASC",
        (label,),
    ).fetchall()
    out = []
    for o in rows:
        out.append({
            "object_id": int(o[0]), "asset_id": int(o[1]), "score": float(o[2]),
            "bbox": {"x": int(o[3]), "y": int(o[4]), "w": int(o[5]), "h": int(o[6])},
            "cluster_id": int(o[7]) if o[7] is not None else None,
        })
    return out


def scene_nearest(store: IndexStore, query_vec, k: int = 10) -> list[dict]:
    """Scene search: nearest-K (K=10 fixed) over scene_vec by cosine distance;
    results are asset rows, not people (§8 Scene search)."""
    q = np.asarray(query_vec, dtype=np.float32).ravel()
    rows = store.conn.execute(
        "SELECT s.asset_id, s.embedding, a.path FROM scene_vec s"
        " JOIN assets a ON a.id = s.asset_id"
    ).fetchall()
    if not rows:
        return []
    ids = [int(r[0]) for r in rows]
    cands = np.vstack([blob_to_vec(bytes(r[1])) for r in rows])
    dists = 1.0 - (cands @ q)
    order = np.argsort(dists)[:k]
    return [
        {"asset_id": ids[i], "distance": float(dists[i]), "path": rows[i][2]}
        for i in order
    ]


def faces_for_asset(store: IndexStore, asset_id: int) -> list[dict]:
    rows = store.conn.execute(
        "SELECT id, cluster_id FROM faces WHERE asset_id = ? ORDER BY id ASC", (asset_id,)
    ).fetchall()
    return [{"face_id": int(r[0]), "cluster_id": int(r[1]) if r[1] is not None else None} for r in rows]


def objects_for_asset(store: IndexStore, asset_id: int) -> list[dict]:
    rows = store.conn.execute(
        "SELECT id, cluster_id FROM objects WHERE asset_id = ? ORDER BY id ASC", (asset_id,)
    ).fetchall()
    return [{"object_id": int(r[0]), "cluster_id": int(r[1]) if r[1] is not None else None} for r in rows]


def asset_path(store: IndexStore, asset_id: int) -> str | None:
    row = store.conn.execute("SELECT path FROM assets WHERE id = ?", (asset_id,)).fetchone()
    return row[0] if row is not None else None
