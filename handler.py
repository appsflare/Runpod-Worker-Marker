"""
RunPod serverless worker for Marker PDF conversion.

Input schema (job["input"]):
    pdf             - Required. Base64-encoded PDF/image bytes, or a URL to download the file from.
    filename        - Optional. Original filename (used for extension detection). Defaults to "document.pdf".
    page_range      - Optional. Page range string, e.g. "0-5". Defaults to all pages.
    force_ocr       - Optional. Force OCR even if text layer exists. Defaults to False.
    paginate_output - Optional. Add page delimiters to output. Defaults to False.
    output_format   - Optional. One of: "markdown", "json", "html", "chunks". Defaults to "markdown".
    mode            - Optional. Conversion mode: "balanced" or "fast".
                      "balanced" (the default on GPU) uses the surya VLM for layout and
                      full-page OCR - required for scanned/image-only documents.
                      "fast" uses the lightweight rf-detr layout detector and only
                      block-OCRs garbled or empty content. Cheaper, but not a substitute
                      for OCR on scanned pages.
                      When omitted, Marker picks by device (balanced on CUDA).
    use_llm         - Optional. Enable LLM-assisted conversion. Defaults to False.
    llm_service     - Optional. Fully-qualified LLM service class path.
                      Defaults to "marker.services.ollama.OllamaService".
                      Only used when use_llm=True.
    llm_config      - Optional. Dict of service-specific config passed directly to the service
                      constructor (e.g. {"ollama_model": "qwen3-vl:8b",
                      "ollama_base_url": "http://localhost:11434"}).
                      Only used when use_llm=True - the keys are ignored otherwise.
    action          - Optional. Control message for the worker lifecycle.
                      "stop_ollama": gracefully stop the background Ollama server.
                      When set, no PDF conversion is performed.

Output schema:
    success         - True on successful conversion.
    filename        - Original filename.
    output_format   - Format used for conversion.
    markdown        - Markdown text (when output_format="markdown").
    html            - HTML text (when output_format="html").
    json            - Structured JSON dict (when output_format="json").
    chunks          - Chunks text (when output_format="chunks").
    images          - Dict of image name -> base64-encoded image string, encoded using
                      Marker's OUTPUT_IMAGE_FORMAT setting (JPEG by default, not PNG).
                      Populated for non-JSON output formats; empty for output_format="json".
    metadata        - Marker metadata dict.
    page_count      - Number of pages processed.

Environment variables:
    SURYA_INFERENCE_BACKEND
                    - "llamacpp" or "vllm". marker 2.0 serves layout/OCR from a VLM behind
                      an OpenAI-compatible endpoint rather than loading torch models in
                      process. Surya autodetects "vllm" whenever a GPU is present, and that
                      backend spawns its server with `docker run` - impossible in a RunPod
                      serverless container. The Dockerfile pins this to "llamacpp", which
                      spawns the llama-server binary directly. Do not unset it.
    SURYA_GGUF_LOCAL_MODEL_PATH / SURYA_GGUF_LOCAL_MMPROJ_PATH
                    - Paths to the VLM weights baked into the image. Setting both makes
                      surya skip its HuggingFace download at runtime.
    MODEL_CACHE_DIR - Where surya's s3:// checkpoints (text detection, OCR error) live.
    HF_HOME         - Where the GGUF and hf:// weights live. MODEL_CACHE_DIR does NOT
                      cover these; both must point at persistent storage if you swap the
                      baked-in models for a network volume.
    TORCH_DEVICE    - Device for surya's auxiliary torch servers: text detection, OCR-error
                      and fast-layout. It does NOT control the OCR/layout VLM, which runs
                      under llama-server and is GPU-offloaded via LLAMA_CPP_NGL.
                      (What became a no-op in marker 2.0 is the create_model_dict(device=)
                      argument, not this variable - the predictors are thin clients, but
                      the servers they talk to read this setting. Leave it "cuda" on a GPU
                      worker; setting "cpu" silently moves those servers off the GPU.)
"""

import base64
import gc
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import requests
import runpod
from ollama_runner import OllamaRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Every surya/marker setting below is read straight from the environment by a
# pydantic-settings singleton built at *import* time, and marker is imported at
# module scope right below. Anything set from Python after that point is
# ignored, so all of these are configured as Dockerfile ENV instead:
#
#   SURYA_INFERENCE_BACKEND      - must be llamacpp; the GPU autodetect would
#                                  otherwise pick vllm, which spawns via docker
#   SURYA_GGUF_LOCAL_*_PATH      - baked-in VLM weights
#   MODEL_CACHE_DIR / HF_HOME    - two separate caches; see the module docstring
#
# Resist the temptation to os.environ.setdefault() any of them here.

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
VALID_OUTPUT_FORMATS = {"markdown", "json", "html", "chunks"}
VALID_MODES = {"balanced", "fast"}

# ---------------------------------------------------------------------------
# Model loading – executed once when the container starts.
# ---------------------------------------------------------------------------

logger.info("Initializing Marker predictors...")
try:
    from marker.models import create_model_dict

    # Under marker 2.0 this holds no weights - every entry is a thin client of an
    # out-of-process surya server (see marker/models.py). So this call is cheap and
    # proves nothing about whether inference actually works.
    MODELS = create_model_dict()
    logger.info("Marker predictors constructed (%d).", len(MODELS))
except Exception:
    logger.exception("Failed to construct Marker predictors.")
    MODELS = None


def ensure_inference_server() -> Optional[str]:
    """Make sure the surya VLM server is up. Returns None on success, else an error.

    Kept separate from predictor construction on purpose. Surya spawns the server
    lazily on the first layout/OCR call, which would put a multi-GB model load
    inside job #1 where RunPod's per-job timeout applies rather than surya's own
    600s startup budget. So we start it eagerly below.

    But an eager start that fails must not brick the worker: manager.start() is
    idempotent (it returns the existing handle if already running), so calling it
    again on a later job lets a transient failure - a slow volume mount, a port
    still draining from a previous container - heal itself.
    """
    if MODELS is None:
        return "Marker predictors failed to construct. Check container logs."
    manager = MODELS["inference_manager"]
    try:
        manager.start()
        return None
    except Exception as exc:
        logger.exception(
            "surya inference server (backend=%s) failed to start. If this is a "
            "SpawnError the exception text is generic - the real cause is in the "
            "server's own log at ~/.cache/datalab/surya/llamacpp_server.log.",
            manager.method,
        )
        return f"Inference server ({manager.method}) failed to start: {exc}"


_startup_error = ensure_inference_server()
if _startup_error:
    logger.error("Startup: %s -- will retry on the first job.", _startup_error)
else:
    logger.info("Startup: surya inference server ready.")

# ---------------------------------------------------------------------------
# OllamaRunner singleton – shared across all jobs (warm-start reuse).
# ---------------------------------------------------------------------------

ollama_runner = OllamaRunner()


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _resolve_file(pdf_input: str, filename: str) -> bytes:
    """Return raw bytes from a base64 string or a URL."""
    if pdf_input.startswith(("http://", "https://")):
        logger.info("Downloading file from URL: %s", pdf_input)
        response = requests.get(pdf_input, timeout=120)
        response.raise_for_status()
        return response.content
    # Assume base64-encoded bytes
    try:
        return base64.b64decode(pdf_input, validate=True)
    except Exception as exc:
        raise ValueError(f"Invalid base64 input: {exc}") from exc

# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """Process a single PDF conversion job."""
    job_input: dict = job.get("input", {})

    # --- action messages (no pdf conversion needed) ---
    if job_input.get("action") == "stop_ollama":
        ollama_runner.stop()
        return {"success": True, "message": "Ollama server stopped."}

    # --- required field ---
    pdf_input: Optional[str] = job_input.get("pdf")
    if pdf_input is None:
        return {"error": "Missing required field: 'pdf' (base64 string or URL)."}
    pdf_input = pdf_input.strip()
    if not pdf_input:
        return {"success": False, "error": "Missing required field: 'pdf' (base64 string or URL)."}

    # --- optional fields ---
    filename: str = job_input.get("filename", "document.pdf")
    page_range: Optional[str] = job_input.get("page_range")
    force_ocr: bool = bool(job_input.get("force_ocr", False))
    paginate_output: bool = bool(job_input.get("paginate_output", False))
    output_format: str = job_input.get("output_format", "markdown")
    mode: Optional[str] = job_input.get("mode")
    use_llm: bool = bool(job_input.get("use_llm", False))
    llm_service_path: Optional[str] = job_input.get("llm_service")
    llm_config: Optional[dict] = job_input.get("llm_config",{})

    if use_llm and not llm_service_path:
        llm_service_path = "marker.services.ollama.OllamaService"

    if llm_config is not None and not isinstance(llm_config, dict):
        return {"success": False, "error": "'llm_config' must be a JSON object (dict)."}

    # --- validate extension ---
    file_ext = Path(filename).suffix.lower() or ".pdf"
    if file_ext not in ALLOWED_EXTENSIONS:
        return {
            "success": False,
            "error": f"Unsupported file type '{file_ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        }

    # --- validate output format ---
    if output_format not in VALID_OUTPUT_FORMATS:
        return {
            "success": False,
            "error": f"Invalid output_format '{output_format}'. Must be one of: {sorted(VALID_OUTPUT_FORMATS)}",
        }

    # --- validate mode ---
    if mode is not None and mode not in VALID_MODES:
        return {
            "success": False,
            "error": f"Invalid mode '{mode}'. Must be one of: {sorted(VALID_MODES)}",
        }

    # --- inference server must be up before we can convert anything ---
    # Deliberately after input validation: a malformed request should get a
    # message about the request, not about the server.
    server_error = ensure_inference_server()
    if server_error:
        return {"success": False, "error": server_error}

    # --- resolve file bytes ---
    try:
        file_bytes = _resolve_file(pdf_input, filename)
    except Exception as exc:
        logger.exception("Failed to retrieve file.")
        return {"success": False, "error": f"Failed to retrieve file: {exc}"}

    # --- write to a temp file and convert ---
    # temp_path starts as None so the finally clause is safe even if the
    # NamedTemporaryFile context manager raises before the assignment.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.settings import settings

        config = {
            "filepath": temp_path,
            "page_range": page_range,
            "force_ocr": force_ocr,
            "paginate_output": paginate_output,
            "output_format": output_format,
            "use_llm": use_llm,
            "llm_service": llm_service_path,
        }

        # Only merge caller-supplied keys when they are actually going to a service.
        # Splatting unconditionally let any caller write arbitrary top-level Marker
        # config through a field documented as service-specific. Truthiness check
        # rather than `is not None` - callers may send "llm_config": null.
        if use_llm and llm_config:
            config.update(llm_config)

        # After the merge, so an explicit mode wins over a stray llm_config key.
        # Left unset, PdfConverter picks by device: balanced on CUDA, fast otherwise.
        if mode is not None:
            config["mode"] = mode

        config_parser = ConfigParser(config)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1

        # --- lazy-start Ollama if needed ---
        if use_llm and llm_service_path and OllamaRunner.is_ollama_service(llm_service_path):
            _base_url = (llm_config or {}).get("ollama_base_url", "http://localhost:11434")
            _model = (llm_config or {}).get("ollama_model")
            ollama_runner.ensure_ready(_base_url, _model)

        converter = PdfConverter(
            config=config_dict,
            artifact_dict=MODELS,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
            
        )

        logger.info("Converting '%s' to %s …", filename, output_format)
        rendered_output = converter(temp_path)

        # --- extract content ---
        json_content = None
        html_content = None
        markdown_content = None
        chunks_content = None
        encoded_images: dict = {}

        if output_format == "json":
            json_content = rendered_output.model_dump()
        else:
            from marker.output import text_from_rendered

            text, _, images = text_from_rendered(rendered_output)

            if output_format == "html":
                html_content = text
            elif output_format == "chunks":
                chunks_content = json.loads(text)
            else:
                markdown_content = text

            for img_name, img_obj in images.items():
                buf = io.BytesIO()
                img_obj.save(buf, format=settings.OUTPUT_IMAGE_FORMAT)
                encoded_images[img_name] = base64.b64encode(buf.getvalue()).decode("utf-8")

        metadata = rendered_output.metadata

        # Guard against a silent empty result.
        #
        # Marker does not propagate inference failures. LayoutBuilder logs
        # "Layout inference failed for page N; leaving page empty" and carries on
        # (marker/builders/layout.py:231), so if the surya server rejects every
        # request the pipeline still renders a structurally valid document with no
        # content - and we would bill the caller for an empty string while
        # reporting success. That is a worse failure than crashing, because
        # nothing downstream can tell it apart from a genuinely blank page.
        #
        # A page with no blocks has an empty block_counts
        # (marker/renderers/__init__.py:104-117), so "no blocks on any page"
        # catches the whole class: server down mid-job, bad weights, OOM under
        # concurrency.
        #
        # Note this is a backstop for TOTAL failure, not partial. Marker has real
        # fallbacks - if layout inference dies, full-page OCR still rebuilds the
        # page, and on a digital PDF pdftext recovers the text layer - so a
        # partial failure legitimately still produces content and is not caught
        # here. That is the intended behaviour: only "nothing at all" is an error.
        page_stats = metadata.get("page_stats") or []
        if page_stats and not any(p.get("block_counts") for p in page_stats):
            logger.error(
                "Conversion of '%s' produced no blocks on any of %d page(s).",
                filename,
                len(page_stats),
            )
            return {
                "success": False,
                "error": (
                    f"Conversion produced no content across all {len(page_stats)} "
                    "page(s). Either the document is genuinely blank, or layout/OCR "
                    "inference failed for every page - check the worker logs and "
                    "~/.cache/datalab/surya/llamacpp_server.log."
                ),
            }

        logger.info("Conversion of '%s' completed successfully.", filename)

        return {
            "success": True,
            "filename": filename,
            "output_format": output_format,
            "markdown": markdown_content,
            "html": html_content,
            "json": json_content,
            "chunks": chunks_content,
            "images": encoded_images,
            "metadata": metadata,
            "page_count": len(metadata.get("page_stats", [])),
        }

    except Exception as exc:
        logger.exception("Conversion failed for '%s'.", filename)
        return {"success": False, "error": f"Conversion failed: {exc}"}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        gc.collect()


if __name__ == "__main__":
    try:
        runpod.serverless.start({"handler": handler})
    finally:
        # Both servers are subprocesses of this container and outlive the handler
        # unless stopped explicitly.
        ollama_runner.stop()
        if MODELS is not None:
            from marker.models import shutdown_models

            shutdown_models(MODELS)
