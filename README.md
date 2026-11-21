# vision-embed

Two services, one contract each. Nothing here is implemented yet.

vision-embed is a stateless measurement service. It accepts an image and returns face embeddings, object labels, and one scene vector. It does not store photos, vectors, names, or cluster ids.

vision-index is a separate process one operator runs for themselves. It calls vision-embed, stores the vectors in SQLite, clusters faces, and records a name only when the operator assigns one. Another person runs their own index. They do not share a people table.

## Specs

- vision-embed-SPEC.md — DRAFT v4. Measurement service. Build from this file alone.
- vision-index-SPEC.md — DRAFT v4. Identity index. Build from this file alone. It must not contradict the embed spec.

Both were scored by a critic. Features under 85 were patched until the registers and the pair cleared 85. Do not reopen settled decisions in the specs. If a build needs a behavior the spec does not state, stop and ask. Do not invent it.

## Next

1. Review both specs. The drafts are the contract, not a suggestion.
2. When the review is done, create the git repo. Do not create it before that. Do not put personal identifiers in the public repo.
3. Implement vision-embed first, from its spec, including the fake adapter and the tests. Do not download model weights at runtime. Weights are operator-supplied files.
4. Implement vision-index second, against a running embed service or the fake embed adapter. Do not run InsightFace inside the index process.
5. A directory of photos is a caller loop of POST /v1/ingest. Do not add a directory walker unless the spec is revised first.

## Out of scope until a spec change

GUI, MCP server, phone backup, multi-tenant people tables, public unauthenticated access, storing image bytes, CUDA in v1, automatic naming.
