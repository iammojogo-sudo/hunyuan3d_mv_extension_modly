"""
Hunyuan3D-2mv - Modly extension generator.

Pipeline:
  1. Preprocess the uploaded front image and any optional side views.
  2. Run Hunyuan3DDiTFlowMatchingPipeline with front/left/back/right inputs.
  3. Export a GLB mesh to the Modly workspace.
"""
import base64
import io
import os
import sys
import threading
import time
import uuid
from pathlib import Path

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress


# Redirect print to stderr so stdout stays clean for the JSON runner protocol.
_print = print


def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)


_HF_REPO_ID = "tencent/Hunyuan3D-2mv"

_SUBFOLDERS = {
    "hunyuan3d-dit-v2-mv-turbo": "hunyuan3d-dit-v2-mv-turbo",
    "hunyuan3d-dit-v2-mv-fast": "hunyuan3d-dit-v2-mv-fast",
    "hunyuan3d-dit-v2-mv": "hunyuan3d-dit-v2-mv",
}


def _safe_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_bool(val, default=True):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        text = val.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
    if val is None:
        return default
    return bool(val)


def _strip_data_url(value):
    if isinstance(value, str) and "," in value and value[:64].lower().startswith("data:"):
        return value.split(",", 1)[1]
    return value


class Hunyuan3D2mvGenerator(BaseGenerator):
    MODEL_ID = "hunyuan3d2mv"
    DISPLAY_NAME = "Hunyuan3D-2mv"
    VRAM_GB = 8
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv-turbo"

    def is_downloaded(self):
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        marker = self.model_dir / self.MODEL_VARIANT / "model.fp16.safetensors"
        return marker.exists()

    def _ensure_hy3dgen_on_path(self):
        repo_dir = Path(__file__).parent / "Hunyuan3D-2"
        if not repo_dir.exists():
            raise RuntimeError(
                "Hunyuan3D-2 source not found at %s. Please reinstall or repair the extension."
                % repo_dir
            )
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

    def load(self):
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_hy3dgen_on_path()

        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._loaded_variant = None
        self._pipeline = None
        self._torch = torch
        self._model = True

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        import torch

        params = params or {}
        variant = params.get("model_variant") or self.MODEL_VARIANT
        steps = _safe_int(params.get("num_inference_steps"), 30)
        octree_res = _safe_int(params.get("octree_resolution"), 380)

        self._report(progress_cb, 10, "Loading model...")
        self.load()

        self._report(progress_cb, 20, "Generating mesh...")
        stop_evt = threading.Event()
        progress_thread = None
        if progress_cb:
            progress_thread = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 30, 92, "Generating mesh...", stop_evt),
                daemon=True,
            )
            progress_thread.start()

        try:
            generator = torch.Generator(device=self._device).manual_seed(42)
            with torch.no_grad():
                result = self._pipeline(
                    image=image_bytes,
                    num_inference_steps=steps,
                    octree_resolution=octree_res,
                    guidance_scale=5.0,
                    num_chunks=8000,
                    box_v=1.01,
                    mc_level=0.0,
                    generator=generator,
                    output_type="trimesh",
                )
                mesh = result[0]
        finally:
            stop_evt.set()
            if progress_thread:
                progress_thread.join(timeout=1.0)

        self._report(progress_cb, 100, "Done")
        return str(mesh)
