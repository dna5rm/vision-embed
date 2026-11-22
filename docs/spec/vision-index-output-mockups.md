# vision-index — output mockups (agreed 2026-11-21)

Data model: **people** have names; **clusters** are mechanical groupings of similar
faces and have **no name and no identity of their own**; a cluster is *linked* to a
person as evidence of that person, or left unlinked. **faces** carry the vectors and
belong to exactly one cluster.

No `assets` table. `vec_hash` is the face primary key.
No `clusters.name` column — a cluster cannot be named, only linked as evidence.

All numbers below come from the operator's real DB (60 faces of one person from
`~/Pictures/photos_a`, 21 clusters). They are illustrative of shape, not a fixture.

---

## `vision-index list people`

```
$ vision-index list people

PEOPLE
  name        faces  linked clusters  last added
  alice         42                1  2026-11-21
  bob             0                0  -

  1 person. 21 clusters unlinked (60 faces).
```

Notes:
- `linked clusters` column exists deliberately — a person being evidenced by many clusters is normal, not a defect.
- A person with 0 faces is a valid state (created, no evidence yet). `link <cluster> <person>` creates the person if the name is new.

---

## `vision-index list clusters`

```
$ vision-index list clusters

CLUSTERS
  id  person    faces  added
   3  alice       20  2026-11-21
   4  alice       11  2026-11-21
  14  alice        7  2026-11-21
   6  alice        3  2026-11-21
   1  -             1  2026-11-21
   2  -             1  2026-11-21
   5  -             1  2026-11-21
  11  -             1  2026-11-21
   ...

  21 clusters: 4 linked, 17 unlinked
```

Notes:
- **No name column.** The word 'name' does not appear here because a cluster cannot be named — nowhere to put a cluster name, ever.
- `-` in `person` means the cluster is unlinked evidence.

---

## `vision-index status`

```
$ vision-index status

UNLINKED CLUSTERS
  id  faces  nearest person  distance  nearest cluster  distance
   1      1  alice             0.5313  cluster 4          0.4210
   2      1  alice             0.4535  cluster 3          0.3884
  11      1  alice             0.5431  cluster 16         0.3951
   5      1  alice             0.4302  cluster 14         0.3208
   ...

  17 unlinked clusters, 18 faces
  suggested: vision-index link --near alice 0.45
```

Notes:
- **Two distances**: to the nearest person, and to the nearest other cluster. The second
  column is what makes fragmentation visible on the first enrol instead of after 60 photos.
- Ends with a runnable suggestion derived from the measured distances — the suggestion
  links clusters as evidence of a person; it never names anything.
- Read-only: `status` never writes.

---

## `vision-index list faces`

```
$ vision-index list faces

FACES
  vec_hash    cluster  person   det    added
  3f9a2c8b1d  3        alice   0.80   2026-11-21
  a71e40c95f  3        alice   0.76   2026-11-21
  c28d5b01e7  1        -        0.86   2026-11-21
  9e4b7a2f63  2        -        0.68   2026-11-21
  ...

  60 faces, 4 clusters linked, 17 unlinked
```

Notes:
- `vec_hash` truncated to 10 chars in the table view; `--json` carries the full 64.
- Replaces the old `list assets`, which is dropped.

---

## Verb surface after the redesign

| verb | meaning |
|---|---|
| `enroll <photo>` | embed, store faces, cluster them mechanically. Never creates or touches a person. |
| `identify <photo>` | min distance to any face of any cluster linked to each person |
| `link <cluster> <person>` | record that this cluster is evidence of this person (creates the person if the name is new) |
| `link --near <person> <dist>` | link every unlinked cluster within `<dist>` of that person |
| `unlink <cluster>` | the cluster is evidence of nobody |
| `delete cluster <id> --yes` | delete a cluster and its faces |
| `delete person <name> --yes` | delete the person, their linked clusters, and those clusters' faces in ONE transaction (cascade; nothing survives its person) |
| `list people` / `list clusters` / `list faces` | as above |
| `status` | unlinked work queue + suggested command |

Gone: `rename`, `merge`, `attach`, `enroll --add-to`, `assign`, `enroll --person`,
`list assets`, `find object`, `find like`.

---

## Vocabulary rules

- A cluster has no name. Nothing in the tool may name a cluster.
- Naming a person is an explicit human act; `link` never invents a name from a photo.
- `enroll` never creates a person and never links — it only clusters.

---

## Constants

| constant | value | used by |
|---|---|---|
| `FACE_MATCH_MAX` | 0.28 | `identify` only |
| `CLUSTER_JOIN_MAX` | 0.40 | ingest clustering only |
| `FACE_DET_MIN` | 0.65 | ingest clustering only |

No consolidation pass. Fragmenting into many clusters is harmless once clusters are
evidence rather than candidates for being a person.
