"""Pre-download every model the worker needs, at image build time.

Cold starts on RunPod serverless are bounded by the per-job timeout, so a
multi-GB download inside job #1 shows up as a job failure rather than a slow
first request. Everything is fetched here instead.

marker 2.0 / surya 0.22 pull weights from three different places, and only one
of them honours MODEL_CACHE_DIR:

  * the OCR/layout VLM, as GGUF  -> huggingface_hub, cached under HF_HOME
  * the rf-detr fast-layout model -> huggingface_hub (hf:// refs), HF_HOME
  * text detection + ocr error   -> surya's own S3 bucket, MODEL_CACHE_DIR

The GGUFs land in a fixed directory (not the content-addressed HF cache) so the
Dockerfile can point SURYA_GGUF_LOCAL_MODEL_PATH / _MMPROJ_PATH at stable
paths. Setting those makes surya skip hf_hub_download entirely at runtime
(surya/inference/backends/llamacpp.py:106-113), so a cold start does no
network I/O at all.

Run with the same env the container uses -- this reads surya's settings
singleton, so MODEL_CACHE_DIR/HF_HOME must already be exported.
"""

import os
import sys

from huggingface_hub import hf_hub_download
from surya.common.rfdetr_torch import resolve_model_dir
from surya.common.s3 import download_directory
from surya.settings import settings

# Where the two GGUF files go. Must match SURYA_GGUF_LOCAL_*_PATH in the
# Dockerfile; that is what makes the runtime skip the download.
GGUF_DIR = os.environ.get("SURYA_GGUF_DIR", "/models/gguf")


def download_gguf() -> None:
    """Fetch the VLM weights + multimodal projector used by the llamacpp backend."""
    os.makedirs(GGUF_DIR, exist_ok=True)
    for filename in (settings.SURYA_GGUF_MODEL_FILE, settings.SURYA_GGUF_MMPROJ_FILE):
        print(f"[models] {settings.SURYA_GGUF_REPO}/{filename} -> {GGUF_DIR}")
        path = hf_hub_download(
            repo_id=settings.SURYA_GGUF_REPO,
            filename=filename,
            local_dir=GGUF_DIR,
        )
        print(f"[models]   {path} ({os.path.getsize(path) / 1e9:.2f} GB)")


def download_s3_checkpoints() -> None:
    """Fetch surya's s3:// checkpoints into MODEL_CACHE_DIR.

    Mirrors S3DownloaderMixin.get_local_path + from_pretrained
    (surya/common/s3.py:130-166): strip the scheme, join onto the cache dir,
    then download. Doing it the same way means the runtime finds these already
    present and skips the fetch.
    """
    for checkpoint in (
        settings.DETECTOR_MODEL_CHECKPOINT,
        settings.OCR_ERROR_MODEL_CHECKPOINT,
    ):
        remote = checkpoint.replace("s3://", "")
        local = os.path.join(settings.MODEL_CACHE_DIR, remote)
        os.makedirs(local, exist_ok=True)
        print(f"[models] {checkpoint} -> {local}")
        download_directory(remote, local)


def download_fast_layout() -> None:
    """Fetch the rf-detr layout + reading-order models used by mode='fast'."""
    for checkpoint in (
        settings.FAST_LAYOUT_MODEL_CHECKPOINT,
        settings.FAST_ORDER_MODEL_CHECKPOINT,
    ):
        print(f"[models] {checkpoint}")
        print(f"[models]   -> {resolve_model_dir(checkpoint)}")


def main() -> int:
    print(f"[models] MODEL_CACHE_DIR={settings.MODEL_CACHE_DIR}")
    print(f"[models] HF_HOME={os.environ.get('HF_HOME', '<unset>')}")
    download_gguf()
    download_s3_checkpoints()
    download_fast_layout()
    print("[models] all downloads complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
