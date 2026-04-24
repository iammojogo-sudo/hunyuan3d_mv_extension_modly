"""
Hunyuan3D-2mv - Modly extension generator
Reference: https://github.com/Tencent-Hunyuan/Hunyuan3D-2

Pipeline:
  1. Remove background from each provided view image with rembg
  2. Run Hunyuan3DDiTFlowMatchingPipeline with front/left/back/right inputs
  3. Export GLB
"""
import io
import os
import sys

# Redirect print to stderr so stdout stays clean for JSON protocol
_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)
import time
import uuid
import threading
import tempfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_HF_REPO_ID = "tencent/Hunyuan3D-2mv"

_SUBFOLDERS = {
    "hunyuan3d-dit-v2-mv-turbo": "hunyuan3d-dit-v2-mv-turbo",
    "hunyuan3d-dit-v2-mv-fast":  "hunyuan3d-dit-v2-mv-fast",
    "hunyuan3d-dit-v2-mv":       "hunyuan3d-dit-v2-mv",
}


class Hunyuan3D2mvGenerator(BaseGenerator):
    MODEL_ID       = "hunyuan3d2mv"
    DISPLAY_NAME   = "Hunyuan3D-2mv"
    VRAM_GB        = 8
    MODEL_VARIANT  = "hunyuan3d-dit-v2-mv-turbo"

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self):
        marker = self.model_dir / "hunyuan3d-dit-v2-mv" / "model.fp16.safetensors"
        return marker.exists()

    def _ensure_hy3dgen_on_path(self):
        repo_dir = Path(__file__).parent / "Hunyuan3D-2"
        if not repo_dir.exists():
            raise RuntimeError(
                "Hunyuan3D-2 source not found at %s. "
                "Please reinstall the extension." % repo_dir
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
        from hy3dgen.rembg import BackgroundRemover

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._rembg = BackgroundRemover()

        # We load lazily per variant in generate() to allow switching
        # but pre-warm with the default turbo model here.
        self._loaded_variant = None
        self._pipeline = None
        self._torch = torch
        self._Pipeline = Hunyuan3DDiTFlowMatchingPipeline

        self._model = True  # non-None sentinel for BaseGenerator
        print("[Hunyuan3D2mvGenerator] Ready on %s." % device)

    def _load_variant(self, variant):
        if self._loaded_variant == variant:
            return
        import torch
        print("[Hunyuan3D2mvGenerator] Loading variant: %s ..." % variant)
        if self._pipeline is not None:
            del self._pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        subfolder = _SUBFOLDERS.get(variant, "hunyuan3d-dit-v2-mv-turbo")
        self._pipeline = self._Pipeline.from_pretrained(
            str(self.model_dir),
            subfolder=subfolder,
            use_safetensors=True,
            device=self._device,
        )
        self._loaded_variant = variant
        print("[Hunyuan3D2mvGenerator] Variant loaded: %s" % variant)

    def unload(self):
        self._pipeline = None
        self._loaded_variant = None
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes,
        params,
        progress_cb=None,
        cancel_event=None,
    ):
        import torch

        variant        = params.get("model_variant") or self.MODEL_VARIANT
        steps          = int(params.get("num_inference_steps", 30))
        octree_res     = int(params.get("octree_resolution", 380))
        seed           = int(params.get("seed", 42))

        def _decode_param(key):
            val = params.get(key)
            if val is None:
                return None
            if params.get(key + "_is_b64"):
                return base64.b64decode(val)
            return val

        import base64
        print("[Hunyuan3D2mvGenerator] params keys: %s" % list(params.keys()))
        print("[Hunyuan3D2mvGenerator] left_image present: %s, is_b64: %s" % (
            params.get("left_image") is not None, params.get("left_image_is_b64")))
        print("[Hunyuan3D2mvGenerator] back_image present: %s, is_b64: %s" % (
            params.get("back_image") is not None, params.get("back_image_is_b64")))
        print("[Hunyuan3D2mvGenerator] right_image present: %s, is_b64: %s" % (
            params.get("right_image") is not None, params.get("right_image_is_b64")))
        left_bytes     = _decode_param("left_image")
        back_bytes     = _decode_param("back_image")
        right_bytes    = _decode_param("right_image")

        # -- Background removal & preprocessing ---------------------------
        self._report(progress_cb, 5, "Preprocessing front view...")
        front_image = self._preprocess_bytes(image_bytes)
        self._check_cancelled(cancel_event)

        image_dict = {"front": front_image}

        if left_bytes:
            self._report(progress_cb, 10, "Preprocessing left view...")
            image_dict["left"] = self._preprocess_bytes(left_bytes)
            self._check_cancelled(cancel_event)

        if back_bytes:
            self._report(progress_cb, 14, "Preprocessing back view...")
            image_dict["back"] = self._preprocess_bytes(back_bytes)
            self._check_cancelled(cancel_event)

        if right_bytes:
            self._report(progress_cb, 18, "Preprocessing right view...")
            image_dict["right"] = self._preprocess_bytes(right_bytes)
            self._check_cancelled(cancel_event)

        print("[Hunyuan3D2mvGenerator] image_dict keys: %s" % list(image_dict.keys()))
        for k, v in image_dict.items():
            print("[Hunyuan3D2mvGenerator] image_dict[%s] type=%s" % (k, type(v).__name__))
        # -- Load variant -------------------------------------------------
        self._report(progress_cb, 22, "Loading model variant...")
        self._load_variant(variant)
        self._check_cancelled(cancel_event)

        # -- Shape generation ---------------------------------------------
        self._report(progress_cb, 30, "Generating mesh...")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 30, 92, "Generating mesh...", stop_evt),
                daemon=True,
            )
            t.start()

        # -- Debug: patch prepare_image to log cond_inputs ------------------
        _orig_prepare = self._pipeline.prepare_image
        def _debug_prepare(image):
            result = _orig_prepare(image)
            print("[Hunyuan3D2mvGenerator] prepare_image output keys: %s" % list(result.keys()))
            for k, v in result.items():
                if hasattr(v, 'shape'):
                    print("[Hunyuan3D2mvGenerator] cond_inputs[%s] shape=%s" % (k, list(v.shape)))
                else:
                    print("[Hunyuan3D2mvGenerator] cond_inputs[%s] value=%s" % (k, v))
            return result
        self._pipeline.prepare_image = _debug_prepare

        try:
            generator = torch.Generator(device=self._device).manual_seed(seed)
            with torch.no_grad():
                mesh = self._pipeline(
                    image=image_dict,
                    num_inference_steps=steps,
                    octree_resolution=octree_res,
                    num_chunks=20000,
                    generator=generator,
                    output_type="trimesh",
                )[0]
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # -- Export GLB ---------------------------------------------------
        self._report(progress_cb, 94, "Exporting GLB...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name     = "%d_%s.glb" % (int(time.time()), uuid.uuid4().hex[:8])
        out_path = self.outputs_dir / name
        mesh.export(str(out_path))

        self._report(progress_cb, 100, "Done")
        return out_path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess_bytes(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._remove_bg(img)

    def _preprocess_path(self, path):
        img = Image.open(path).convert("RGB")
        return self._remove_bg(img)

    def _remove_bg(self, img):
        try:
            result = self._rembg(img)
            return result
        except Exception:
            # Fall back: return image as-is if rembg fails
            return img

    def _download_weights(self):
        from huggingface_hub import snapshot_download
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3D2mvGenerator] Downloading weights from %s ..." % _HF_REPO_ID)
        snapshot_download(
            repo_id=_HF_REPO_ID,
            local_dir=str(self.model_dir),
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print("[Hunyuan3D2mvGenerator] Weights downloaded.")

@classmethod
    def params_schema(cls):
        return [
            {
                "id": "model_variant",
                "label": "Model Variant",
                "type": "select",
                "default": "hunyuan3d-dit-v2-mv-turbo",
                "options": [
                    {"value": "hunyuan3d-dit-v2-mv-turbo", "label": "hunyuan3d-dit-v2-mv-turbo"},
                    {"value": "hunyuan3d-dit-v2-mv-fast", "label": "hunyuan3d-dit-v2-mv-fast"},
                    {"value": "hunyuan3d-dit-v2-mv", "label": "hunyuan3d-dit-v2-mv"},
                ],
                "tooltip": "Model variant (locked per node).",
            },
            {
                "id": "num_inference_steps",
                "label": "Inference Steps",
                "type": "select",
                "default": 30,
                "options": [
                    {"value": 10, "label": "Fast (10)"},
                    {"value": 30, "label": "Balanced (30)"},
                    {"value": 50, "label": "Quality (50)"},
                ],
                "tooltip": "Number of diffusion steps.",
            },
            {
                "id": "octree_resolution",
                "label": "Mesh Resolution",
                "type": "select",
                "default": 380,
                "options": [
                    {"value": 256, "label": "Low (256)"},
                    {"value": 380, "label": "Medium (380)"},
                    {"value": 512, "label": "High (512)"},
                ],
                "tooltip": "Octree resolution. Higher = more detail but more VRAM.",
            },
            {
                "id": "guidance_scale",
                "label": "Guidance Scale",
                "type": "select",
                "default": "5.0",
                "options": [
                    {"value": "1.0", "label": "1.0 — loose"},
                    {"value": "3.0", "label": "3.0"},
                    {"value": "5.0", "label": "5.0 — default"},
                    {"value": "7.5", "label": "7.5"},
                    {"value": "10.0", "label": "10.0 — tight"},
                ],
                "tooltip": "Classifier-free guidance strength. Higher = closer to input image.",
            },
            {
                "id": "num_chunks",
                "label": "Decode Chunks",
                "type": "select",
                "default": 8000,
                "options": [
                    {"value": 2000, "label": "Low (2000) — less VRAM"},
                    {"value": 8000, "label": "Medium (8000)"},
                    {"value": 20000, "label": "High (20000) — faster"},
                ],
                "tooltip": "VAE decode chunk size. Lower saves VRAM; higher is faster.",
            },
            {
                "id": "box_v",
                "label": "Bounding Box Scale",
                "type": "select",
                "default": "1.01",
                "options": [
                    {"value": "0.75", "label": "0.75 — small"},
                    {"value": "1.01", "label": "1.01 — default"},
                    {"value": "1.25", "label": "1.25 — large"},
                    {"value": "1.5", "label": "1.5 — extra large"},
                ],
                "tooltip": "Bounding box scale for mesh extraction. Increase if mesh edges are being clipped.",
            },
            {
                "id": "mc_level",
                "label": "Surface Level",
                "type": "select",
                "default": "0.0",
                "options": [
                    {"value": "-0.05", "label": "-0.05 — thicker"},
                    {"value": "-0.02", "label": "-0.02"},
                    {"value": "0.0", "label": "0.0 — default"},
                    {"value": "0.02", "label": "0.02"},
                    {"value": "0.05", "label": "0.05 — thinner"},
                ],
                "tooltip": "Marching cubes iso-surface level. Increase to thin the mesh; decrease to thicken it.",
            },
            {
                "id": "remove_bg",
                "label": "Remove Background",
                "type": "select",
                "default": "true",
                "options": [
                    {"value": "true", "label": "Yes — auto remove background"},
                    {"value": "false", "label": "No — images already masked"},
                ],
                "tooltip": "Run rembg on input images. Disable if your images already have a transparent background.",
            },
            {
                "id": "seed",
                "label": "Seed",
                "type": "int",
                "default": 42,
                "min": 0,
                "max": 4294967295,
                "tooltip": "Change if result is unsatisfying.",
            },
        ]
