# Runpod-Worker-Marker

A [RunPod](https://www.runpod.io/) serverless worker that converts PDFs and images to Markdown, HTML, JSON, or chunks using [Marker](https://github.com/datalab-to/marker).

---

## Features

- Converts PDF, PNG, JPEG, TIFF, and BMP files.
- Output formats: `markdown`, `html`, `json`, `chunks`.
- Optional LLM-assisted conversion with **your choice of LLM service and model**.
- GPU-accelerated via CUDA.
- Dependency management via **[UV](https://github.com/astral-sh/uv)**.

---

## CI/CD — Build & Push to GitHub Container Registry

A GitHub Actions workflow (`.github/workflows/docker-build-push.yml`) automatically builds and pushes the Docker image to the GitHub Container Registry (`ghcr.io`) on every push to `main` and on semver tags (`v*.*.*`).

Authentication uses the built-in `GITHUB_TOKEN` — **no extra secrets or variables are required**. The image is published under `ghcr.io/<owner>/<repo>` (e.g. `ghcr.io/appsflare/runpod-worker-marker`).

### Tags produced

| Trigger | Tags pushed |
|---------|-------------|
| Push to `main` | `:latest`, `:sha-<short-sha>` |
| Tag `v1.2.3` | `:v1.2.3`, `:1.2`, `:sha-<short-sha>` |

---

## Local Development

### Prerequisites

- [UV](https://docs.astral.sh/uv/getting-started/installation/) installed.
- Python 3.11+.
- (Optional) CUDA-capable GPU.

### Setup

```bash
uv sync
```

This reads `pyproject.toml` / `uv.lock` and installs all dependencies (including PyTorch with CUDA 13.0 wheels, from the `pytorch-cu130` index).

> `uv.lock` is committed and the Docker build installs with `--frozen`. Change dependencies through `pyproject.toml` + `uv lock`, never by editing the lockfile — an unpinned build is how the worker silently crossed the Marker 1.x → 2.0 boundary.

---

## Building the Docker Image

```bash
docker build -t runpod-worker-marker:latest .
```

---

## Deploying on RunPod

1. Push the image to a container registry (Docker Hub, GHCR, etc.).
2. Create a new **Serverless** endpoint in the RunPod console.
3. Set the container image to your pushed image.
4. Configure the following environment variables as needed:

| Variable | Default | Description |
|----------|---------|-------------|
| `SURYA_INFERENCE_BACKEND` | `llamacpp` | Which server serves the OCR/layout VLM. **Do not unset** — see below. |
| `LLAMA_CPP_BINARY` | `/opt/llama.cpp/llama-server` | Absolute path to the server binary. |
| `SURYA_GGUF_LOCAL_MODEL_PATH` | `/models/gguf/surya-2.gguf` | VLM weights baked into the image. |
| `SURYA_GGUF_LOCAL_MMPROJ_PATH` | `/models/gguf/surya-2-mmproj.gguf` | Multimodal projector baked into the image. |
| `SURYA_INFERENCE_KEEP_ALIVE` | `True` | Leave `llama-server` running between jobs so warm workers re-attach instead of reloading the model. |
| `MODEL_CACHE_DIR` | `/models` | Surya's `s3://` checkpoints (text detection, OCR error). |
| `HF_HOME` | `/models/hf` | The GGUF and `hf://` weights. **`MODEL_CACHE_DIR` does not cover these.** |
| `TORCH_DEVICE` | `cuda` | Device for Surya's **auxiliary** torch servers (text detection, OCR-error, fast layout). Does *not* control the OCR/layout VLM — that runs under `llama-server`, offloaded via `LLAMA_CPP_NGL`. Leave it `cuda` on a GPU worker. |

### The inference backend

Marker 2.0 (Surya ≥ 0.22) no longer runs layout and OCR as in-process PyTorch models. They are served by a VLM (`datalab-to/surya-ocr-2`) behind an OpenAI-compatible HTTP endpoint, and Surya chooses the server by autodetect: **NVIDIA GPU → `vllm`, otherwise → `llamacpp`**.

The `vllm` backend starts its server by shelling out to `docker run`. A RunPod serverless container has no Docker daemon and no privileged mode, so on a GPU worker the default autodetect fails every job with:

```
surya.inference.backends.spawn.SpawnError: docker binary not found.
```

The image therefore pins `SURYA_INFERENCE_BACKEND=llamacpp`, which spawns the `llama-server` binary (copied from `ghcr.io/ggml-org/llama.cpp:server-cuda13`) as a plain subprocess. `handler.py` starts that server during container init rather than letting Surya spawn it lazily, so a slow model load surfaces as a startup failure instead of eating a job's timeout.

**If a conversion fails with a `SpawnError`**, the exception text is generic — the real cause is in the server's own log at `~/.cache/datalab/surya/llamacpp_server.log`.

> If you change where `llama-server` is copied to, keep the `ldconfig` step next to it in the `Dockerfile`. The executable is a small launcher that dynamically links ~150MB of sibling `.so` files, and its `RUNPATH` points at a build-time path that does not exist in the image — without the `ld.so.conf.d` entry it fails at exec with `error while loading shared libraries`, which surfaces only as a generic `SpawnError`.

### Why guided decoding is disabled

`SURYA_GUIDED_LAYOUT=False` is set in the image because guided decoding is **pure cost** on this backend.

Surya constrains layout output with a JSON schema whose `bbox` field carries the regex `^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$`. llama.cpp compiles response schemas into GBNF grammars, and its converter cannot handle that pattern — `llama-server` answers *every* layout request with `HTTP 400 "Failed to initialize samplers: failed to parse grammar"`. Bisected against llama.cpp `b10156`: the `bbox` pattern is the sole trigger; `maxItems` and the integer bounds compile fine.

The failure is quiet rather than fatal, which is why it is easy to miss. Surya retries 4× per page, logs `Layout inference failed for page N; leaving page empty`, and continues:

- **balanced mode** — full-page OCR rebuilds the page afterwards, so the text still comes out correct. You just burn four failed round-trips per page and lose the layout model's contribution.
- **pdftext path** — structure degrades instead. In testing a heading came back as plain `Quarterly Report` rather than `## Quarterly Report`.

Disabling guided decoding is a supported Surya mode, not a workaround — the sibling `SURYA_GUIDED_TABLE_REC` already ships `False` by default, noting that the model produces well-formed JSON without a schema. The cost is occasional malformed-JSON retries.

This is specific to llama.cpp's grammar converter, so a future `llama.cpp` bump may fix it. To re-test: set `SURYA_GUIDED_LAYOUT=True` and watch for HTTP 400s in `~/.cache/datalab/surya/llamacpp_server.log`. vLLM handles the full schema, so this constraint disappears on that backend.

Separately, `handler.py` rejects any conversion that yields no blocks on *any* page, returning an explicit error rather than an empty document. Marker does not propagate inference failures — it logs them per page and still renders — so this is a backstop against a total backend failure being billed as a successful conversion of nothing.

### Tuning throughput

`SURYA_INFERENCE_PARALLEL` is intentionally left unset. `llama-server` sizes its KV cache as `max(16384, parallel × SURYA_INFERENCE_CTX_PER_SLOT)` with a 12288 per-slot default, so raising parallelism multiplies context demand — 32 slots would request ~393k. Surya's own default of 8 is the safe starting point.

If you raise it, watch the startup log. When effective per-slot context drops below 12288, Surya logs a warning and `llama-server` then **silently truncates OCR output** rather than erroring — it looks like a bad PDF, not a crash. Raise `SURYA_INFERENCE_CTX_SIZE` to compensate.

For higher throughput than llama.cpp provides, Surya can attach to an external vLLM server via `SURYA_INFERENCE_URL` (which skips the Docker spawn entirely). That server must advertise the model as exactly `datalab-to/surya-ocr-2` and answer `/health`, or Surya rejects it.

### Using a persistent volume for models

All weights are baked into the image, so cold starts do no network I/O. To use a **Network Volume** instead, attach it in the RunPod console and set **both** `MODEL_CACHE_DIR` *and* `HF_HOME` to paths under its mount point (e.g. `/runpod-volume/models` and `/runpod-volume/models/hf`). Setting only `MODEL_CACHE_DIR` leaves the VLM and layout weights downloading to the container's ephemeral filesystem on every cold start.

---

## Input Schema

Send a JSON payload to your endpoint:

| Field            | Type    | Required | Default      | Description                                                                                         |
|------------------|---------|----------|--------------|-----------------------------------------------------------------------------------------------------|
| `pdf`            | string  | ✅       | —            | Base64-encoded file bytes **or** a public URL to download the file from.                            |
| `filename`       | string  | ❌       | `document.pdf` | Original filename; used for file-type detection.                                                  |
| `page_range`     | string  | ❌       | all pages    | Page range, e.g. `"0-5"`.                                                                           |
| `force_ocr`      | boolean | ❌       | `false`      | Force OCR even when a text layer is present.                                                        |
| `paginate_output`| boolean | ❌       | `false`      | Insert page delimiters into the output.                                                             |
| `output_format`  | string  | ❌       | `"markdown"` | One of `"markdown"`, `"html"`, `"json"`, `"chunks"`.                                               |
| `mode`           | string  | ❌       | by device    | `"balanced"` or `"fast"`. See below.                                                                |
| `use_llm`        | boolean | ❌       | `false`      | Enable LLM-assisted conversion.                                                                     |
| `llm_service`    | string  | ❌       | `"marker.services.ollama.OllamaService"` | Fully-qualified LLM service class. Requires `use_llm=true`. |
| `llm_config`     | object  | ❌       | —            | Service-specific config dict passed to the service constructor. **Ignored unless `use_llm=true`.** See examples below. |

### Conversion mode

| Mode | Layout | OCR | Use when |
|------|--------|-----|----------|
| `balanced` | Surya VLM | Full-page VLM OCR | Default on GPU. **Required for scanned / image-only PDFs.** |
| `fast` | Lightweight rf-detr detector | Block-OCRs only garbled or empty content | Digital PDFs with a good text layer, when you want lower cost per page. |

Omit the field to let Marker choose by device (`balanced` on CUDA). `fast` is a cost lever, not a substitute for OCR — it will not read a scanned page.

### Example — Markdown (base64 input)

```json
{
  "input": {
    "pdf": "<base64-encoded PDF bytes>",
    "filename": "report.pdf",
    "output_format": "markdown"
  }
}
```

### Example — HTML via URL

```json
{
  "input": {
    "pdf": "https://example.com/document.pdf",
    "filename": "document.pdf",
    "output_format": "html"
  }
}
```

### Example — LLM-assisted with Ollama (qwen3-vl:8b)

```json
{
  "input": {
    "pdf": "<base64-encoded PDF bytes>",
    "filename": "report.pdf",
    "output_format": "markdown",
    "use_llm": true,
    "llm_service": "marker.services.ollama.OllamaService",
    "llm_config": {
      "ollama_model": "qwen3-vl:8b",
      "ollama_base_url": "http://localhost:11434"
    }
  }
}
```

---

## Output Schema

| Field          | Type            | Description                                                     |
|----------------|-----------------|-----------------------------------------------------------------|
| `success`      | boolean         | `true` on successful conversion.                                |
| `filename`     | string          | Original filename.                                              |
| `output_format`| string          | Format used.                                                    |
| `markdown`     | string \| null  | Markdown text (when `output_format="markdown"`).                |
| `html`         | string \| null  | HTML text (when `output_format="html"`).                        |
| `json`         | object \| null  | Structured data (when `output_format="json"`).                  |
| `chunks`       | string \| null  | Chunks text (when `output_format="chunks"`).                    |
| `images`       | object          | Map of image name → base64-encoded image, using Marker's `OUTPUT_IMAGE_FORMAT` (JPEG by default). |
| `metadata`     | object          | Marker metadata (language, page stats, etc.).                   |
| `page_count`   | integer         | Number of pages processed.                                      |
| `error`        | string          | Present only on failure; describes what went wrong.             |

