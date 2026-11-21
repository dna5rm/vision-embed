# vision-embed: additive CUDA spec (CUDAExecutionProvider)

Status: DRAFT v1, revised. Additive only. This document modifies nothing in `vision-embed-SPEC.md` (the v4 contract stays as-is); it defines the changes for enabling `CUDAExecutionProvider`, the fail-closed gates around it, and the measurement work that has been recorded. A CUDA deployment is authorized under this spec: the code may enable `CUDAExecutionProvider`, and startup is fatal if the provider is missing or fails to activate — no silent CPU fallback. The live default remains CPU.

## 1. Executive locks (do not reopen)

1. **No numeric CPU-vs-CUDA envelope exists, and none is invented here.** The measurement method is defined here; no acceptance envelope has been accepted, so no per-provider numeric equality or bitwise-equality claim is made anywhere in either direction. Numeric equality with CPU is not certified, and CUDA is never to be called production-equal while no envelope exists.
2. **Detector geometry and embedding vectors are separate measurements.** Bboxes, det scores, and kps are measured apart from embedding vectors, with no shared envelope.
3. **No new fields in the HTTP success schema or the existing audit line.** The closed `additionalProperties: false` schema and the audit line format are untouched. Provider visibility is limited to a single startup log line, or nothing.
4. **One image.** The existing `nvidia/cuda:12.6.3-runtime-ubuntu24.04` base (pinned by digest) stays. The image changes are exactly two: swapping the pip pin from `onnxruntime` to the matching CUDA 12 `onnxruntime-gpu` wheel, and adding the cuDNN libraries that wheel requires (the supply mechanism remains open item 1). Nothing else in the image changes. `CPUExecutionProvider` remains valid on that image (the GPU wheel's CPU provider works without a GPU or the nvidia container runtime). No second image variant.
5. **Live default stays CPU, CUDA is authorized.** The running default remains `CPUExecutionProvider`. A CUDA deployment IS authorized under this spec — no interim study-only restriction applies. The configuration is fail-closed: startup fatals if the provider is missing (lock 6).
6. **Fail-fatal, never fallback.** If `CUDAExecutionProvider` is configured and the provider is not registered in the onnxruntime build or does not activate on a visible device, startup is fatal `config.invalid`: exactly one JSON line on stdout, exit 1, no listener, no metrics port, no CPU fallback, no partial start.
7. **No hostnames, personal names, or IPs** anywhere in this document, its configs, or its test fixtures.

## 2. Current state (verified inputs)

- `create_session(path, provider)` in `vision_embed/adapters/onnx.py` is the only place an `ort.InferenceSession` is built; all four adapter roles (face embedder, face detector, object detector, scene embedder) call it. It currently rejects anything but `CPUExecutionProvider` with a ValueError, then asserts the built session reports that provider first (`RuntimeError("onnxruntime did not activate ...")`). Both checks stay structurally; the allowlist widens.
- Config: `models.execution_provider`, string, under the closed `_MODELS_KEYS` set, default `"CPUExecutionProvider"`. Validation in `vision_embed/config.py` currently rejects any other value with `FatalStartupError("config.invalid", ...)`.
- Dockerfile: `nvidia/cuda:12.6.3-runtime-ubuntu24.04` base pinned by amd64 manifest digest; CPU `onnxruntime==1.20.1` pip pin; the Dockerfile comment already reserves the swap. The base ships libcudart and the CUDA math libraries; it does not ship cuDNN; onnxruntime-gpu requires cuDNN libraries to be supplied in the image at build time. The cuDNN supply mechanism is an open item (section 11), and a missing cuDNN library is a startup fatal, not a surprise after T-C7 passes. The base runs under plain runc with no GPU.
- The image comment notes the CPU wheel loads CUDA libraries lazily only if CUDAExecutionProvider is requested; on the swapped GPU wheel the CPU provider still loads and runs without a visible GPU, so the unchanged CPU default keeps working on the new image. This property is tested below (section 7), not assumed.

## 3. Config key

- Key: `models.execution_provider`, exact string, unchanged location and closed-schema status.
- Post-flip allowed values, an explicit two-value allowlist:
  - `"CPUExecutionProvider"` (default; unchanged behavior)
  - `"CUDAExecutionProvider"` (opt-in; fatal at startup if unavailable, per lock 6)
- Any other string (including `TensorrtExecutionProvider`, `azure`, or an "auto" value) stays fatal `config.invalid` from config validation, exactly as today.
- Config validator change is shape only: replace the `!= "CPUExecutionProvider"` rejection with membership in the two-value allowlist; same fatal path, same `config.invalid` token, same one-line stdout format.
- No fallback chaining (e.g. `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`) anywhere. create_session always builds with a single-element provider list so failure is explicit.

## 4. Startup order (for `execution_provider: "CUDAExecutionProvider"`)

Checked in this order, all before any listener or metrics port binds:

1. **Config validation.** Value must be exactly one of the two allowlist entries. Anything else: fatal `config.invalid` during config parse (existing path).
2. **Provider-registered pre-flight.** Before loading any weights, check `"CUDAExecutionProvider"` is present in `ort.get_available_providers()`. If absent (CPU-only or wrongly-built wheel), fatal `config.invalid` with detail naming the provider and the likely cause (onnxruntime-gpu not installed). This is one clean error instead of four identical session errors. It must emit exactly one required startup log line on success (the wording in lock 3 permits no other provider-related logging; the log line itself is mandatory, not optional, and T-C6 asserts it); nothing else provider-related is logged.
3. **Per-session activation gate.** For each of the four sessions, `create_session` builds with `providers=["CUDAExecutionProvider"]`; if construction raises, or `session.get_providers()[0] != "CUDAExecutionProvider"`, startup fatals. A session that constructs successfully while CPU leads the active provider list (subgraph exclusions, unsupported kernels) is exactly as fatal as a missing device. If any one of the four sessions fails, the whole process fatals; no per-model CPU fallback.
4. **No app-code probing.** Do not check `/dev/nvidia*`, `nvidia-smi` output, CUDA driver versions, or device inventories. onnxruntime's activation check is the ground truth. No retry, no poll, no re-check, no marker file, no network egress to verify drivers.

## 5. Fatal shape (unchanged contract)

- Exactly one JSON line on stdout: `{"fatal": true, "reason": "config.invalid", "detail": "<why>"}`, then exit 1, before any listener or metrics port opens.
- The reason stays `config.invalid`. No new token for "driver missing" vs "provider not compiled in" vs "session did not activate"; the distinction lives in `detail` only. Detail is a single line, safe for logs, and carries no image or token content.
- No HTTP 503 for a missing GPU. 503 `not_ready` means "models still loading"; a CUDA-selected process that cannot get CUDA sessions never serves any request and never opens a socket.
- The closed nine-token reason set does not grow.

## 6. Dockerfile and compose change

- Dockerfile: more than one edit. Replace the pip pin `onnxruntime==1.20.1` with the matching `onnxruntime-gpu` 1.20.x GPU wheel pinned to an exact patch version (e.g. `onnxruntime-gpu==1.20.1`; exact patch confirmed against the wheel index before merge). The image must also carry whatever cuDNN version the chosen `onnxruntime-gpu` wheel requires; the supply mechanism is open item 1, decided before merge. A missing or wrong-version cuDNN library surfaces as a startup fatal (section 5), not as a silent CPU mode. Update the lines 28 to 29 comment to reflect the swap and the cuDNN supply. Same base digest, same ENTRYPOINT, same user, same read-only posture, same EXPOSE.
- Version coupling to record explicitly: the `onnxruntime-gpu` wheel must be compatible with the pinned base's CUDA 12.x runtime libraries and the cuDNN version it requires. Verify the wheel's required CUDA runtime and cuDNN versions against the base and the image contents before merging. This is the external compatibility check the change introduces, and it covers cuDNN, not CUDA alone.
- `CPUExecutionProvider on the GPU-wheel image` is a supported state: the GPU wheel's CPU provider runs with no visible GPU and without the nvidia container runtime. Test CI-D (section 7) proves it on the built image. With the CPU default, a plain-runc host behaves exactly as today.
- Compose (deploy concern, not the Dockerfile): the GPU path requires an nvidia device reservation, either a commented-out `gpus: all` shorthand or a `deploy.resources.reservations.devices` block (driver `nvidia`, count `all`, capabilities `["gpu"]`) at the location the compose comments already mark. Its exact shape is an open item (section 11). Without a reservation, startup fatals by section 4, which is intended behavior, not an accident. No base-image variant per host class; the host's runtime does the differentiation (lock 4).

## 7. Tests

### GPU-free tests (CPU CI)

- **T-C1 config allowlist:** `CUDAExecutionProvider` parses into config; an arbitrary string stays fatal `config.invalid`; default remains `CPUExecutionProvider`. Inverts the existing v1-fatality test.
- **T-C2 create_session allowlist:** `provider="CUDAExecutionProvider"` passes the allowlist; the ValueError path only fires for other strings. Test notes skip any activation assertion when the environment actually has a working CUDA provider.
- **T-C3 activation failure to fatal:** on a CPU-only environment, building with `providers=["CUDAExecutionProvider"]` either raises or reports a different provider first; create_session must turn whichever occurs into its existing activation RuntimeError, i.e. no silent fallback. Skip this test if the environment really has CUDA.
- **T-C4 fake-adapter plumbing:** run the full endpoint suite with `adapter: "fake"` under both config values on a CPU host; proves config plumbing reaches all four adapter constructors without sessions being built.
- **T-C5 startup error content:** production-mode startup with `CUDAExecutionProvider` on a CPU-only image asserts fatal error code `config.invalid`, one JSON line, exit before bind, and detail naming the provider.
- **T-C6 image check (CI):** assert the built image's venv contains `onnxruntime-gpu` and not `onnxruntime` (pip freeze check); assert the `ort.get_available_providers()` pre-flight log line appears at boot; assert the CPU provider reports available on the same image.
- **T-C7 CPU default intact:** boot the GPU-wheel image with the default CPU config under plain runc (no GPU reservation); startup succeeds and CPU sessions activate. Locks down "CPU stays valid on that image".

### GPU-only tests (requires nvidia container runtime and a reserved device)

- **T-G1 activation:** production mode with all four real weights and `CUDAExecutionProvider`; startup succeeds; each session's actual provider equals `CUDAExecutionProvider` (observed via the boot log line).
- **T-G2 fallback-fatalism:** same GPU image without the compose device reservation; startup must fatal, never serve on CPU.
- **T-G3 OOM behavior:** one image at large resolution to confirm GPU memory exhaustion is a clean fatal error, not a hang.
- **T-G4 latency (informational, not a gate):** timed detect/embed calls CPU vs CUDA on a fixed image set, reported, not asserted, until the measured delta justifies any throughput claim.

## 8. Tolerance study (measurement recorded; no envelope invented)

Purpose: record the CPU-vs-CUDA measurement distributions per lock 2. This is a measurement, not a gate: it neither limits nor authorizes the deployment (lock 5) and it produces no acceptance envelope. No acceptance envelope exists anywhere under this spec.

Fixture set: several images covering plain, borderline low-res, multi-face, no-face, and dense-object cases, run through both providers on the same machine, same pinned weights, same onnxruntime minor version, fixed input bytes (identical preprocessing path by construction; preprocessing is pure numpy and provider-independent).

Embedding measurements, per model and per vector family (512-d face, 768-d scene, 768-d object crops), on all four models (det_10g, w600k_r50, openai-ViT-L-14-visual, yolov8n as applicable to each family):

- max absolute error, element-wise;
- max relative error, element-wise;
- cosine distance after the existing L2-normalize step.

Detector measurements, kept separate per lock 2, on detector outputs (bboxes, det scores, kps):

- max absolute error on box coordinates and scores;
- box IoU between the CPU and CUDA detections for matched pairs, plus a count of split matches (detections present on one side only), since small numeric drift can flip a face across the score threshold.

Also recorded: a restart-and-swap check (same bytes, CPU run vs CUDA run vs CUDA run after restart, all inside whatever envelope the measurements support, should an envelope ever be accepted), and full per-session provider confirmation during the runs.

Output: a written report with the measured distributions. The measured distributions are cited as measurements only. A measured CPU-vs-CUDA study has been recorded: face embeddings max abs ~4.95e-04 difference with cosine similarity ~0.9999945; scene embeddings max abs ~7.72e-05. These numbers are descriptive measurements, not an accepted envelope, and they certify nothing. No per-family numeric envelope has been accepted or proposed here; inventing one is forbidden (lock 1).

## 9. Explicit non-goals

- No implementation in this task; no code, config, image, or compose edits by the author of this spec.
- No CUDA enablement by the author of this spec document; enablement is an implementation concern, authorized under lock 5 and gated by section 4.
- No edit to `vision-embed-SPEC.md`; its INVs, F-items, and test list are untouched here (additive only). Wording changes there (e.g. scoping the vector tolerance claim per provider) land with the implementation, not with this document.
- No numeric tolerance value, invented or otherwise (lock 1).
- No shared detector-geometry/embedding envelope (lock 2).
- No field additions to the HTTP success schema or the audit line; no response-visible provider label (lock 3).
- No second Dockerfile variant, no multi-stage GPU/CPU image split (lock 4).
- No fallback or fallback chaining, at config validation, session build, or inference time (lock 6).
- No "auto" provider value, no per-model provider overrides, no new session options, no device-id map, no warmup call. The image does carry extra runtime libraries beyond the wheel swap: whatever cuDNN version the chosen `onnxruntime-gpu` wheel requires must be present in the image (section 6).
- Unchanged out-of-scope list from v4: face NMS changes, any vector index, runtime weight downloads.
- No bitwise CPU-vs-CUDA equality claim in either direction.
- No hostnames, personal names, or IPs (lock 7).

## 10. Implementation checklist

These steps are already landed in the tree; this list is the record of what the change required, not a future authorization.

1. Widen the config validator allowlist (section 3).
2. Add the provider-registered pre-flight plus its one required log line (section 4, step 2).
3. Widen the create_session first-line allowlist; keep the single-provider list and the activation gate (section 4, step 3).
4. Swap the Dockerfile pip pin with an exact verified patch version, and supply the cuDNN libraries the wheel requires per open item 1 (section 6).
5. The compose GPU block: for a non-study host the example stays commented; a study host may enable it (section 6, open item).
6. Land GPU-free tests T-C1 through T-C7; stand up the GPU-only test environment T-G1 through T-G4.
7. Run the tolerance study per section 8; report the measured distributions. No envelope was accepted, so no amendment pinning numbers exists and none is to be invented.

## 11. Open items (for later decisions)

| # | Item | Held why |
|---|------|----------|
| 1 | Exact `onnxruntime-gpu` patch version, its CUDA 12.x compatibility confirmation, and the cuDNN supply mechanism for the image (which package, build stage, or copy source provides the cuDNN version the wheel requires) | Needs a wheel-index check and cuDNN source decision at implementation time |
| 2 | Compose device block final shape (`gpus: all` vs deploy reservations) | Compose-version dependent |
| 3 | The pre-flight startup log line's required status | Decided here as required, carried into section 4 and T-C6; no longer open |
| 4 | Whether a measured tolerance envelope is ever accepted per model per vector family | No envelope exists; the section 8 study recorded measurements only, and none were blessed as a gate |
| 5 | Driver/CTK version floor pin, if any | Optional; mismatches are already startup-fatal a floor only avoids repeated fatals |
