FROM ollama/ollama:0.19.0 AS ollama

FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

# --------------------------------------------------------------------------- #
# System dependencies
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        git \
        curl \
        wget \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Install Ollama (copied from official image to avoid install script / systemd)
# --------------------------------------------------------------------------- #
COPY --from=ollama /bin/ollama /usr/local/bin/ollama
COPY --from=ollama /lib/ollama /lib/ollama

# --------------------------------------------------------------------------- #
# Install llama-server (surya's inference backend)
#
# marker 2.0 / surya 0.22 no longer run layout+OCR as in-process torch models.
# They talk to a VLM over an OpenAI-compatible HTTP endpoint, and surya picks
# the server backend by autodetect: NVIDIA GPU -> vllm, else -> llamacpp
# (surya/inference/__init__.py:62-68). The vllm backend serves the model by
# shelling out to `docker run` (surya/inference/backends/vllm.py:139-198),
# which cannot work here -- a RunPod serverless container has no docker daemon
# and no privileged mode. The llamacpp backend instead spawns this binary
# directly via subprocess.Popen, needing no daemon at all.
# See SURYA_INFERENCE_BACKEND below.
#
# server-cuda13 is the CUDA 13 build, matching this base image.
#
# Copy the WHOLE directory, not just the executable: /app/llama-server is a
# 17KB launcher that dynamically links libllama-server-impl.so, libggml-cuda.so
# (~150MB of kernels), libmtmd.so and friends, all sitting beside it.
#
# Those libraries are NOT found automatically once relocated -- the binaries
# carry RUNPATH=/app/build/bin, a build-time path that exists nowhere. Without
# the ldconfig registration below, llama-server dies at exec with
# "error while loading shared libraries: libllama-server-impl.so", which surya
# surfaces only as a generic SpawnError.
#
# Verified against this base image: after ldconfig the sole unresolved symbol
# is libcuda.so.1, which the NVIDIA container runtime injects from the host
# driver. CUDA sonames are major-versioned, so the image's CUDA 13.3 build
# links cleanly against this CUDA 13.0.2 base. libggml-cuda.so ships SASS for
# sm_86/89/120/121 -- sm_120 covers the Blackwell RTX PRO 6000.
#
# LLAMA_CPP_BINARY is an absolute path rather than a PATH symlink: surya checks
# os.path.isfile() before shutil.which() (llamacpp.py:33-46).
# --------------------------------------------------------------------------- #
COPY --from=ghcr.io/ggml-org/llama.cpp:server-cuda13 /app/ /opt/llama.cpp/
RUN echo "/opt/llama.cpp" > /etc/ld.so.conf.d/llama-cpp.conf && ldconfig
ENV LLAMA_CPP_BINARY=/opt/llama.cpp/llama-server

# --------------------------------------------------------------------------- #
# Install UV
# --------------------------------------------------------------------------- #
COPY --from=ghcr.io/astral-sh/uv:0.5.9 /uv /usr/local/bin/uv

# MODEL_CACHE_DIR and HF_HOME must be set before the model download below --
# they decide where it writes. The rest of the surya settings are runtime-only
# and are deliberately declared *after* the expensive layers, so tuning them
# doesn't invalidate the dependency install and re-download every model.
#
# HF_HOME matters: MODEL_CACHE_DIR only covers surya's s3:// checkpoints. The
# GGUF and hf:// weights go through huggingface_hub, which ignores it.
ENV UV_SYSTEM_PYTHON=1 \
    UV_PYTHON=python3.12 \
    TORCH_DEVICE=cuda \
    MODEL_CACHE_DIR=/models \
    HF_HOME=/models/hf \
    PYTHONUNBUFFERED=1

# --------------------------------------------------------------------------- #
# Install Python dependencies via UV
#
# uv.lock is copied and --frozen enforced so the build installs exactly the
# resolved versions. Without the lockfile the build re-resolves every time,
# which is how marker silently crossed the 1.x -> 2.0 boundary.
# --------------------------------------------------------------------------- #
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project --frozen

# --------------------------------------------------------------------------- #
# Pre-download models so cold starts do no network I/O
# --------------------------------------------------------------------------- #
COPY download_models.py ./
RUN /app/.venv/bin/python download_models.py

# --------------------------------------------------------------------------- #
# Surya inference backend (runtime settings; kept after the heavy layers)
#
# SURYA_INFERENCE_BACKEND overrides the GPU autodetect that would otherwise
# select vllm-via-docker. The GGUF paths point at the files baked in above; when
# both are set surya uses them verbatim and skips hf_hub_download entirely
# (surya/inference/backends/llamacpp.py:106-113), so a cold start does no
# network I/O. KEEP_ALIVE lets a recycled-but-warm worker re-attach to an
# already-running llama-server via surya's sentinel instead of respawning and
# reloading the model.
#
# Deliberately NOT setting SURYA_INFERENCE_PARALLEL. llama-server sizes its KV
# cache as max(16384, parallel * SURYA_INFERENCE_CTX_PER_SLOT) with a 12288
# per-slot default (llamacpp.py:119-123), so raising parallel multiplies the
# ctx demand -- 32 slots would ask for 393k. The backend's own default of 8 is
# the safe starting point. Tune only against the startup log; if effective
# per-slot ctx falls below 12288, llama-server silently TRUNCATES OCR output
# instead of erroring (llamacpp.py:129-135).
#
# SURYA_GUIDED_LAYOUT=False because guided decoding is pure cost on this
# backend. Surya's LAYOUT_JSON_SCHEMA constrains bbox with the regex
# ^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$ (surya/inference/prompts.py:121-137) and
# sends it as an OpenAI response_format json_schema. llama.cpp compiles that
# schema to a GBNF grammar and its converter cannot handle the pattern, so
# llama-server answers EVERY layout request with
#     HTTP 400 "Failed to initialize samplers: failed to parse grammar"
# Bisected against llama.cpp b10156: the bbox pattern is the sole trigger;
# maxItems and the integer bounds compile fine.
#
# Measured impact (not fatal, which is why it is easy to miss): surya retries
# 4x per page, logs "Layout inference failed for page N; leaving page empty",
# and carries on. In balanced mode full-page OCR then rebuilds the page, so the
# text still comes out right - you just burn 4 failed round-trips per page and
# lose the layout model's contribution. On the pdftext path the structure
# degrades instead: a heading came back as plain "Quarterly Report" rather than
# "## Quarterly Report".
#
# Turning guided decoding off is a first-class surya mode, not a hack -- the
# sibling SURYA_GUIDED_TABLE_REC already ships False by default with the note
# that "the model produces well-formed JSON without the schema". Revisit if
# moving to vLLM, which does support the full schema.
# --------------------------------------------------------------------------- #
ENV SURYA_INFERENCE_BACKEND=llamacpp \
    SURYA_INFERENCE_KEEP_ALIVE=True \
    SURYA_GUIDED_LAYOUT=False \
    SURYA_GGUF_LOCAL_MODEL_PATH=/models/gguf/surya-2.gguf \
    SURYA_GGUF_LOCAL_MMPROJ_PATH=/models/gguf/surya-2-mmproj.gguf

# --------------------------------------------------------------------------- #
# Copy worker source
# --------------------------------------------------------------------------- #
COPY handler.py openai_service.py ollama_runner.py test_input.json ./

# Call the venv interpreter directly. `uv run` would re-check (and can
# re-resolve) the environment on every container start, adding a network
# dependency to the cold-start path.
CMD ["/app/.venv/bin/python", "-u", "handler.py"]
