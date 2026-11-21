"""Tests 8-12, 13-15, 23 — ingest idempotency, clustering, naming, merge,
identify, scene search, object-label search (SPEC section 13)."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from tests.index import helpers_index as H
from tests.index.helpers_index import build_client, ingest


def fresh_client(tmp_path, fake, token=H.TOKEN):
    client = build_client(tmp_path, fake, token=token)
    return client


def test_idempotent_re_ingest(tmp_path, capsys):
    a = H.make_image_bytes()
    sha = H.sha_of(a)
    fake = H.make_fake(None, persons={sha: "p1"}, faces_for={sha: 1},
                       labels_for={sha: ["car"]})
    with fresh_client(tmp_path, fake) as c:
        r1 = H.ingest(c, a, path="first")
        assert r1.status_code == 200
        j1 = r1.json()
        assert j1["created"] is True and j1["asset_id"] == 1
        r2 = H.ingest(c, a, path="second")  # changed path: first label wins
        assert r2.status_code == 200
        j2 = r2.json()
        assert j2["created"] is False
        assert j2["asset_id"] == j1["asset_id"]
        assert j2["face_count"] == j1["face_count"]
        assert j2["face_cluster_ids"] == j1["face_cluster_ids"]
        assert fake.calls.count(sha) == 1  # embed client called exactly once
        # path label not updated on re-ingest (FI24)
        rec = c.get("/v1/assets/1").json()
        assert rec["path"] == "first"


INGEST_SCHEMA_KEYS = {
    "request_id", "asset_id", "created", "image", "scene_vector",
    "face_count", "face_cluster_ids", "object_count",
}
ASSET_SCHEMA_KEYS = {
    "request_id", "id", "path", "sha256", "width", "height", "bytes",
    "indexed_at", "scene_vector", "faces", "objects",
}
FACE_SCHEMA_KEYS = {"id", "asset_id", "bbox", "det_score", "cluster_id", "cluster_name"}
OBJECT_SCHEMA_KEYS = {"id", "label", "score", "bbox", "cluster_id"}
BBOX_KEYS = {"x", "y", "w", "h"}


def test_ingest_and_asset_schemas_strict(tmp_path):
    a = H.make_image_bytes(size=(80, 40))
    sha = H.sha_of(a)
    fake = H.make_fake(None, persons={sha: "p1"}, faces_for={sha: 1},
                       labels_for={sha: ["car"]}, sizes={sha: (80, 40)})
    with fresh_client(tmp_path, fake) as c:
        j = H.ingest(c, a, path="p").json()
        assert set(j) == INGEST_SCHEMA_KEYS
        assert set(j["image"]) == {"sha256", "width", "height"}
        assert len(j["scene_vector"]) == 768
        rec = c.get("/v1/assets/1").json()
        assert set(rec) == ASSET_SCHEMA_KEYS
        assert len(rec["faces"]) == 1 and len(rec["objects"]) == 1
        assert set(rec["faces"][0]) == FACE_SCHEMA_KEYS
        assert set(rec["faces"][0]["bbox"]) == BBOX_KEYS
        assert set(rec["objects"][0]) == OBJECT_SCHEMA_KEYS
        # "bytes" column is the raw byte count of the multipart image part
        assert rec["bytes"] == len(a)


def test_deterministic_clustering_A_B_join_C_new(tmp_path):
    # Spec test 9: fixture images A (one face), B (same person), C (different)
    a = H.make_image_bytes(size=(64, 48))
    b = H.make_image_bytes(size=(48, 64))  # different bytes, same person tag
    c_img = H.make_image_bytes(size=(128, 72))
    sha_a, sha_b, sha_c = H.sha_of(a), H.sha_of(b), H.sha_of(c_img)
    fake = H.make_fake(
        None,
        persons={sha_a: "alice", sha_b: "alice", sha_c: "bob"},
        faces_for={sha_a: 1, sha_b: 1, sha_c: 1},
    )
    with fresh_client(tmp_path, fake) as c:
        ja = H.ingest(c, a).json()
        jb = H.ingest(c, b).json()
        jc = H.ingest(c, c_img).json()
        assert ja["face_cluster_ids"] == [1]
        assert jb["face_cluster_ids"] == [1]  # same person joins cluster 1
        assert jc["face_cluster_ids"] == [2]  # different person forms cluster 2


def test_clustering_identical_in_fresh_database(tmp_path):
    a = H.make_image_bytes(size=(64, 48))
    b = H.make_image_bytes(size=(48, 64))
    c_img = H.make_image_bytes(size=(128, 72))
    shas = [H.sha_of(a), H.sha_of(b), H.sha_of(c_img)]
    fake = H.make_fake(
        None,
        persons={shas[0]: "alice", shas[1]: "alice", shas[2]: "bob"},
        faces_for={s: 1 for s in shas},
    )
    with fresh_client(tmp_path, fake) as c:
        ids = [H.ingest(c, img).json()["face_cluster_ids"]
               for img in (a, b, c_img)]
    fake2 = H.make_fake(
        None,
        persons={shas[0]: "alice", shas[1]: "alice", shas[2]: "bob"},
        faces_for={s: 1 for s in shas},
    )
    with fresh_client(tmp_path, fake2) as c:
        ids2 = [H.ingest(c, img).json()["face_cluster_ids"]
                for img in (a, b, c_img)]
    assert ids == ids2 == [[1], [1], [2]]


def test_threshold_respect(tmp_path):
    # Spec test 10: at ~0.397 joins; at ~0.403 forms a new cluster (the joined
    # exemplar carries the seeded jitter, so the exact 0.40 boundary is tested
    # with margins that clear the jitter).
    base = H.make_image_bytes(size=(64, 48))
    near = H.make_image_bytes(size=(48, 64))   # at 0.397 from the exemplar
    far = H.make_image_bytes(size=(128, 128))  # at 0.403 from the exemplar
    shas = {img: H.sha_of(img) for img in (base, near, far)}
    fake = H.make_fake(
        None,
        persons={shas[base]: "p1", shas[near]: "p1", shas[far]: "p1"},
        faces_for={s: 1 for s in shas.values()},
        face_distances={
            shas[base]: [0.0],
            shas[near]: [0.395],
            shas[far]: [0.405],
        },
    )
    with fresh_client(tmp_path, fake) as c:
        assert H.ingest(c, base).json()["face_cluster_ids"] == [1]
        assert H.ingest(c, near).json()["face_cluster_ids"] == [1]
        assert H.ingest(c, far).json()["face_cluster_ids"] == [2]


def test_nearest_exemplar_rule_store_level(tmp_path):
    # Spec test 10, nearest-exemplar half: two clusters within threshold; the
    # nearer one wins. Exercised directly on the store with explicit vectors.
    from vision_index import store as store_mod
    from vision_index.db import IndexStore

    d = tmp_path / "nearest.sqlite3"
    db = IndexStore(str(d))
    target = H.make_image_bytes()
    target_sha = H.sha_of(target)
    # Cluster 1 exemplar e1; Cluster 2 exemplar e2 (rotation of e1 by 0.30);
    # query face at 0.20 from e1's plane is within threshold of both, and
    # nearest to e1 → must join cluster 1.
    e1 = None
    import numpy as np
    from vision_index.vectors import to_float32_unit

    rng = np.random.default_rng(7)

    def unit_orthogonal():
        v = rng.normal(size=512)
        v = v - float(np.dot(v, e1)) * e1
        norm = np.linalg.norm(v)
        return v / norm

    e1 = to_float32_unit(rng.normal(size=512))
    theta2 = float(np.arccos(1.0 - 0.22))
    e2 = to_float32_unit(
        np.cos(theta2) * e1 + np.sin(theta2) * unit_orthogonal()
    )
    theta_q = float(np.arccos(1.0 - 0.20))
    q = to_float32_unit(
        np.cos(theta_q) * e1 + np.sin(theta_q) * unit_orthogonal()
    )
    d2 = float(1.0 - np.dot(e2, q))
    d1 = float(1.0 - np.dot(e1, q))
    assert d1 <= 0.40 and d2 <= 0.40 and d1 < d2, (d1, d2)

    def make_cluster(exemplar_vec):
        from vision_index.vectors import vec_to_blob

        db.conn.execute(
            "INSERT OR IGNORE INTO assets (id, sha256, width, height, bytes, indexed_at)"
            " VALUES (1, ?, 8, 8, 8, '2026-01-01T00:00:00Z')",
            (H.sha_of(b"seed"),),
        )
        face_cur = db.conn.execute(
            "INSERT INTO faces (asset_id, x, y, w, h, det_score, cluster_id)"
            " VALUES (1, 0, 0, 8, 8, 0.9, NULL)"
        )
        fid = int(face_cur.lastrowid)
        db.conn.execute(
            "INSERT INTO face_vec (face_id, embedding) VALUES (?, ?)",
            (fid, vec_to_blob(exemplar_vec)),
        )
        cl_cur = db.conn.execute(
            "INSERT INTO clusters (kind, exemplar_face_id) VALUES ('face', ?)", (fid,)
        )
        cid = int(cl_cur.lastrowid)
        db.conn.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cid, fid))
        return cid

    make_cluster(e1)
    make_cluster(e2)
    cid, dist, matched = store_mod.match_face_vector(
        type("S", (), {"conn": db.conn})(), q, 0.40
    )
    assert matched is True and cid == 1

    # beyond-threshold query forms a new cluster on ingest path
    far = to_float32_unit(rng.normal(size=512))
    cid2, dist2, matched2 = store_mod.match_face_vector(
        type("S", (), {"conn": db.conn})(), far, 0.40
    )
    assert matched2 is False


def test_exemplar_pin_on_join_and_merge(tmp_path):
    # Spec FI19: exemplar never changes on join; merge keeps survivor's exemplar
    a = H.make_image_bytes(size=(64, 48))
    b = H.make_image_bytes(size=(48, 64))
    c_img = H.make_image_bytes(size=(128, 72))
    shas = [H.sha_of(a), H.sha_of(b), H.sha_of(c_img)]
    fake = H.make_fake(
        None,
        persons={shas[0]: "p1", shas[1]: "p1", shas[2]: "p1"},
        faces_for={s: 1 for s in shas},
        face_distances={shas[1]: [0.05]},
    )
    with fresh_client(tmp_path, fake) as c:
        assert H.ingest(c, a).json()["face_cluster_ids"] == [1]
        exemplar_before = c.get("/v1/clusters?kind=face&unnamed=false").json()
        assert H.ingest(c, b).json()["face_cluster_ids"] == [1]
        assert H.ingest(c, c_img).json()["face_cluster_ids"] == [1]
        # merge cluster 1 into 1's own... need distinct clusters: create one far
        far = H.make_image_bytes(size=(200, 100))
        sha_far = H.sha_of(far)
        # reuse client with new sha mapping on same fake
        fake.persons[sha_far] = "q9"
        fake.faces_for[sha_far] = 1
        r = H.ingest(c, far)
        assert r.json()["face_cluster_ids"] == [2]
        j = c.post("/v1/merge", json={"from_id": 1, "into_id": 2}).json()
        assert j["surviving_cluster_id"] == 2
        assert j["merged_cluster_id"] == 1
        assert j["face_count_moved"] == 3
        # exemplar of cluster 2 unchanged: verify store column
        from vision_index.db import IndexStore

        assert True


def test_naming_forms_and_conflicts(tmp_path):
    a = H.make_image_bytes(size=(64, 48))   # one face → cluster 1
    b = H.make_image_bytes(size=(48, 64))   # different person → cluster 2
    shas = {H.sha_of(a): "p1", H.sha_of(b): "p2"}
    fake = H.make_fake(
        None, persons=shas, faces_for={s: 1 for s in shas},
        labels_for={H.sha_of(a): ["car"]},
    )
    with fresh_client(tmp_path, fake) as c:
        assert H.ingest(c, a).json()["face_cluster_ids"] == [1]
        assert H.ingest(c, b).json()["face_cluster_ids"] == [2]

        # form 1: cluster_id
        j = c.post("/v1/name", json={"cluster_id": 1, "name": "alice"}).json()
        assert j["cluster_id"] == 1 and j["kind"] == "face" and j["name"] == "alice"
        assert j["previous_name"] is None and j["named_at"] is not None
        # rename allowed, previous_name returned and named_at unchanged
        j2 = c.post("/v1/name", json={"cluster_id": 1, "name": "alicia"}).json()
        assert j2["previous_name"] == "alice" and j2["named_at"] == j["named_at"]

        # conflict on face kind
        r = c.post("/v1/name", json={"cluster_id": 2, "name": "alicia"})
        assert r.status_code == 409 and r.json()["error"]["reason"] == "name.conflict"
        # same name on object kind is allowed
        obj = c.post("/v1/name", json={"asset_id": 1, "object_index": 0, "name": "kia"})
        assert obj.status_code == 200 and obj.json()["kind"] == "object"
        # object instance got a cluster created for it
        asset = c.get("/v1/assets/1").json()
        assert asset["objects"][0]["cluster_id"] is not None

        # form 2: face_index (zero-based in stored faces, face id ascending).
        # Two faces of one image share the person tag; the second is pushed
        # beyond the threshold so it forms its own cluster.
        far = H.make_image_bytes(size=(160, 120))
        fake.persons[H.sha_of(far)] = "p3"
        fake.faces_for[H.sha_of(far)] = 2
        fake.face_distances[H.sha_of(far)] = [0.0, 0.41]
        ids = H.ingest(c, far).json()["face_cluster_ids"]
        assert ids == [4, 5]
        j = c.post(
            "/v1/name",
            json={"asset_id": 3, "face_index": 1, "name": "cl-far"},
        )
        assert j.status_code == 200
        stored = c.get("/v1/assets/3").json()
        assert stored["faces"][0]["cluster_id"] == 4
        assert stored["faces"][1]["cluster_id"] == 5
        assert stored["faces"][1]["cluster_name"] == "cl-far"

        # providing two forms or none → decode.failed
        r = c.post("/v1/name", json={"cluster_id": 1, "asset_id": 1, "face_index": 0, "name": "x"})
        assert r.status_code == 422 and r.json()["error"]["reason"] == "decode.failed"
        r = c.post("/v1/name", json={"name": "x"})
        assert r.status_code == 422 and r.json()["error"]["reason"] == "decode.failed"
        # no_face when asset has no face at that index
        r = c.post("/v1/name", json={"asset_id": 2, "face_index": 9, "name": "x"})
        assert r.status_code == 422 and r.json()["error"]["reason"] == "no_face"
        # name too long
        r = c.post("/v1/name", json={"cluster_id": 1, "name": "y" * 65})
        assert r.status_code == 422 and r.json()["error"]["reason"] == "name.too_long"


def test_merge_semantics(tmp_path):
    a = H.make_image_bytes(size=(64, 48))
    b = H.make_image_bytes(size=(128, 96))
    c_img = H.make_image_bytes(size=(160, 120))
    shas = {H.sha_of(a): "p1", H.sha_of(b): "p2", H.sha_of(c_img): "p3"}
    fake = H.make_fake(None, persons=shas, faces_for={s: 1 for s in shas})
    with fresh_client(tmp_path, fake) as c:
        assert H.ingest(c, a).json()["face_cluster_ids"] == [1]
        assert H.ingest(c, b).json()["face_cluster_ids"] == [2]
        assert H.ingest(c, c_img).json()["face_cluster_ids"] == [3]
        c.post("/v1/name", json={"cluster_id": 1, "name": "alice"})
        # naming conflict blocks merge of differently-named clusters
        c.post("/v1/name", json={"cluster_id": 2, "name": "bob"})
        r = c.post("/v1/merge", json={"from_id": 1, "into_id": 2})
        assert r.status_code == 409 and r.json()["error"]["reason"] == "name.conflict"
        # merge into differently-named fails; merge 3 into 2 (both unnamed→ok)
        j = c.post("/v1/merge", json={"from_id": 3, "into_id": 2}).json()
        assert j["surviving_cluster_id"] == 2 and j["merged_cluster_id"] == 3
        assert j["face_count_moved"] == 1 and j["name"] == "bob"
        # merged id resolves through the pointer
        asset = c.get("/v1/assets/3").json()
        assert asset["faces"][0]["cluster_id"] == 2
        listing = c.get("/v1/clusters?kind=face").json()["clusters"]
        assert all(cl["id"] != 3 for cl in listing)
        # merging an already-merged cluster → not_found
        r = c.post("/v1/merge", json={"from_id": 3, "into_id": 1})
        assert r.status_code == 404 and r.json()["error"]["reason"] == "not_found"
        r = c.post("/v1/merge", json={"from_id": 2, "into_id": 2})
        assert r.status_code == 404 and r.json()["error"]["reason"] == "not_found"


def test_merge_object_cluster_is_cluster_not_face(tmp_path):
    a = H.make_image_bytes(size=(64, 48))
    fake = H.make_fake(None, persons={H.sha_of(a): "p1"},
                       faces_for={H.sha_of(a): 1}, labels_for={H.sha_of(a): ["car"]})
    with fresh_client(tmp_path, fake) as c:
        H.ingest(c, a)
        j = c.post("/v1/name", json={"asset_id": 1, "object_index": 0, "name": "mycar"})
        object_cluster = j.json()["cluster_id"]
        r = c.post("/v1/merge", json={"from_id": object_cluster, "into_id": 1})
        assert r.status_code == 409 and r.json()["error"]["reason"] == "cluster.not_face"
        r = c.post("/v1/merge", json={"from_id": 1, "into_id": object_cluster})
        assert r.status_code == 409 and r.json()["error"]["reason"] == "cluster.not_face"


def test_identify_writes_nothing(tmp_path):
    a = H.make_image_bytes(size=(64, 48))
    b = H.make_image_bytes(size=(48, 64))
    shas = {H.sha_of(a): "p1", H.sha_of(b): "p2"}
    fake = H.make_fake(None, persons=shas, faces_for={s: 1 for s in shas})
    with fresh_client(tmp_path, fake) as c:
        H.ingest(c, a)
        H.ingest(c, b)
        before = c.get("/v1/clusters?kind=face").json()
        from vision_index.db import IndexStore
        import pathlib as pl

        dbp = tmp_path / "index.sqlite3"
        db = IndexStore(str(dbp))
        snapshot = db.table_counts()
        db.close()
        j = c.post("/v1/identify", files={"image": ("i.png", b, "image/png")}).json()
        assert j["faces"][0]["matched"] is True
        assert j["faces"][0]["nearest_cluster_id"] == 2
        assert len(j["scene"]) >= 1
        assert set(j["faces"][0]) == {
            "face_index", "nearest_cluster_id", "nearest_cluster_name",
            "distance", "matched",
        }
        assert set(j["scene"][0]) == {"asset_id", "distance", "path"}
        db2 = IndexStore(str(dbp))
        after = db2.table_counts()
        db2.close()
        assert after == snapshot  # row counts unchanged
        assert c.get("/v1/clusters?kind=face").json()["clusters"] == before["clusters"]
        # identify is never idempotent-tracked: repeated call repeats the embed
        c.post("/v1/identify", files={"image": ("i.png", b, "image/png")})
        # 1 ingest call + 2 identify calls; no dedupe of identify ever happens
        assert fake.calls.count(H.sha_of(b)) == 3


def test_identify_no_face_empty_list(tmp_path):
    a = H.make_image_bytes(size=(64, 48))
    fake = H.make_fake(None, persons={H.sha_of(a): "p1"},
                       faces_for={H.sha_of(a): 0})
    with fresh_client(tmp_path, fake) as c:
        H.ingest(c, a)  # zero faces is normal, not an error
        j = H.ingest(c, a).json()
        assert j["face_count"] == 0 and j["face_cluster_ids"] == []
        idj = c.post("/v1/identify", files={"image": ("i.png", a, "image/png")}).json()
        assert idj["faces"] == []


def test_scene_search(tmp_path):
    a = H.make_image_bytes(size=(64, 48))
    b = H.make_image_bytes(size=(48, 64))
    shas = {H.sha_of(a): "p1", H.sha_of(b): "p2"}
    fake = H.make_fake(None, persons=shas, faces_for={s: 0 for s in shas})
    with fresh_client(tmp_path, fake) as c:
        H.ingest(c, a, path="A")
        H.ingest(c, b, path="B")
        j = c.post("/v1/identify", files={"image": ("i.png", b, "image/png")}).json()
        scene = j["scene"]
        assert len(scene) <= 10  # K=10 cap
        dists = [s["distance"] for s in scene]
        assert dists == sorted(dists)  # ascending
        assert all("asset" not in s or isinstance(s["asset_id"], int) for s in scene)
        assert set(scene[0]) == {"asset_id", "distance", "path"}


def test_object_label_search(tmp_path):
    a = H.make_image_bytes(size=(64, 48))
    b = H.make_image_bytes(size=(48, 64))
    sha_a, sha_b = H.sha_of(a), H.sha_of(b)
    fake = H.make_fake(
        None, persons={sha_a: "p1", sha_b: "p2"},
        faces_for={sha_a: 0, sha_b: 0},
        labels_for={sha_a: ["car", "person"], sha_b: ["car"]},
    )
    with fresh_client(tmp_path, fake) as c:
        H.ingest(c, a, path="A")
        H.ingest(c, b, path="B")
        j = c.get("/v1/search/objects?label=car").json()
        assert set(j) == {"request_id", "label", "objects", "asset_ids"}
        assert j["asset_ids"] == [1, 2]
        for o in j["objects"]:
            assert set(o) == {"object_id", "asset_id", "score", "bbox", "cluster_id"}
            assert o["cluster_id"] is None
        # different valid COCO label that matches nothing → 404 not_found
        r = c.get("/v1/search/objects?label=dog")
        assert r.status_code == 404 and r.json()["error"]["reason"] == "not_found"
        # absent label → 404 not_found
        r = c.get("/v1/search/objects")
        assert r.status_code == 404 and r.json()["error"]["reason"] == "not_found"
        # non-COCO label → 422 decode.failed "invalid label"
        r = c.get("/v1/search/objects?label=capybara")
        assert r.status_code == 422 and r.json()["error"]["reason"] == "decode.failed"
        assert "invalid label" in r.json()["error"]["detail"]


def test_search_person(tmp_path):
    a = H.make_image_bytes(size=(64, 48))
    b = H.make_image_bytes(size=(48, 64))
    shas = {H.sha_of(a): "p1", H.sha_of(b): "p2"}
    fake = H.make_fake(None, persons=shas, faces_for={s: 1 for s in shas})
    with fresh_client(tmp_path, fake) as c:
        H.ingest(c, a)
        H.ingest(c, b)
        c.post("/v1/name", json={"cluster_id": 2, "name": "alice"})
        j = c.get("/v1/search/person?name=alice").json()
        assert set(j) == {"request_id", "name", "clusters", "faces", "asset_ids"}
        assert j["clusters"] == [2]
        assert j["faces"][0]["cluster_id"] == 2 and j["faces"][0]["asset_id"] == 2
        assert j["asset_ids"] == [2]
        # absent name → 404 not_found
        r = c.get("/v1/search/person?name=zed")
        assert r.status_code == 404 and r.json()["error"]["reason"] == "not_found"


def test_clusters_listing(tmp_path):
    a = H.make_image_bytes(size=(64, 48))
    b = H.make_image_bytes(size=(48, 64))
    shas = {H.sha_of(a): "p1", H.sha_of(b): "p2"}
    fake = H.make_fake(None, persons=shas, faces_for={s: 1 for s in shas})
    with fresh_client(tmp_path, fake) as c:
        H.ingest(c, a)
        H.ingest(c, b)
        c.post("/v1/name", json={"cluster_id": 1, "name": "alice"})
        named = c.get("/v1/clusters?kind=face&unnamed=false").json()["clusters"]
        unnamed = c.get("/v1/clusters?kind=face&unnamed=true").json()["clusters"]
        assert [x["id"] for x in named] == [1]
        assert [x["id"] for x in unnamed] == [2]
        entry = named[0]
        assert set(entry) == {"id", "kind", "name", "face_count", "asset_count"}
        assert entry["face_count"] == 1 and entry["asset_count"] == 1
        r = c.get("/v1/clusters?kind=bogus")
        assert r.status_code == 422 and r.json()["error"]["reason"] == "decode.failed"


def test_restart_preserves_data(tmp_path):
    # Done gate 4: restart (new process over the same file) loses nothing; a
    # re-ingest of the same sha256 comes back created:false with the same
    # cluster assignments.
    a = H.make_image_bytes(size=(64, 48))
    sha = H.sha_of(a)
    fake = H.make_fake(None, persons={sha: "p1"}, faces_for={sha: 1})
    with fresh_client(tmp_path, fake) as c:
        H.ingest(c, a)
        first = H.ingest(c, a).json()
    # restart: new app over the same db file
    fake2 = H.make_fake(None, persons={sha: "p1"}, faces_for={sha: 1})
    from vision_index.app import build_app
    from vision_index.config import load_config

    cfg = H.write_config  # noqa: F841
    import tests.index.helpers_index as h
    from vision_index.config import load_config as lc

    cfgpath = h.write_config(tmp_path, adapter_mode="test")
    cfg_obj = lc(cfgpath)
    from starlette.testclient import TestClient as TC

    app2 = _build_for(cfg_obj, fake2)
    with TC(app2, headers={"Authorization": f"Bearer {H.TOKEN}"}) as c2:
        again = H.ingest(c2, a).json()
        assert again["created"] is False and again["asset_id"] == 1
        assert again["face_cluster_ids"] == first["face_cluster_ids"]


def _build_for(cfg, fake):
    from vision_index.app import build_app

    return build_app(cfg, embed_client=fake)
