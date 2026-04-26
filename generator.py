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
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._rembg = BackgroundRemover()
        self._loaded_variant = None
        self._pipeline = None
        self._Pipeline = Hunyuan3DDiTFlowMatchingPipeline
        self._torch = torch
        self._model = True
        print("[Hunyuan3D2mvGenerator] Ready on %s." % self._device)

    def _load_variant(self, variant):
        variant = variant if variant in _SUBFOLDERS else self.MODEL_VARIANT
        if self._loaded_variant == variant:
            return

        import torch

        print("[Hunyuan3D2mvGenerator] Loading variant: %s ..." % variant)
        if self._pipeline is not None:
            del self._pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._pipeline = self._Pipeline.from_pretrained(
            str(self.model_dir),
            subfolder=_SUBFOLDERS[variant],
            use_safetensors=True,
            variant="fp16",
            dtype=self._dtype,
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

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        import torch

        params = params or {}
        variant = params.get("model_variant") or self.MODEL_VARIANT
        steps = _safe_int(params.get("num_inference_steps"), 30)
        octree_res = _safe_int(params.get("octree_resolution"), 380)
        seed = _safe_int(params.get("seed"), 42)
        guidance_scale = _safe_float(params.get("guidance_scale"), 5.0)
        num_chunks = _safe_int(params.get("num_chunks"), 8000)
        box_v = _safe_float(params.get("box_v"), 1.01)
        mc_level = _safe_float(params.get("mc_level"), 0.0)
        remove_bg = _safe_bool(params.get("remove_bg"), True)

        print(
            "[Hunyuan3D2mvGenerator] Parsed params: variant=%s steps=%s octree=%s "
            "guidance=%.2f chunks=%s box_v=%.3f mc_level=%.4f remove_bg=%s seed=%s"
            % (variant, steps, octree_res, guidance_scale, num_chunks, box_v, mc_level, remove_bg, seed)
        )

        self._report(progress_cb, 5, "Preprocessing front view...")
        front_image = self._preprocess_bytes(image_bytes, remove_bg=remove_bg)
        self._check_cancelled(cancel_event)

        image_dict = {"front": front_image}
        for view_name, pct in (("left", 10), ("back", 14), ("right", 18)):
            image = self._optional_view_image(params, view_name, remove_bg)
            if image is None:
                continue
            self._report(progress_cb, pct, "Preprocessing %s view..." % view_name)
            image_dict[view_name] = image
            self._check_cancelled(cancel_event)

        print("[Hunyuan3D2mvGenerator] image_dict keys: %s" % list(image_dict.keys()))

        self._report(progress_cb, 22, "Loading model variant...")
        self._load_variant(variant)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 30, "Generating mesh...")
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
            generator = torch.Generator(device=self._device).manual_seed(seed)
            with torch.no_grad():
                result = self._pipeline(
                    image=image_dict,
                    num_inference_steps=steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=num_chunks,
                    box_v=box_v,
                    mc_level=mc_level,
                    generator=generator,
                    output_type="trimesh",
                )
                print("[Hunyuan3D2mvGenerator] Pipeline result type: %s" % type(result))
                print("[Hunyuan3D2mvGenerator] Pipeline result length: %s" % len(result) if hasattr(result, '__len__') else "N/A")
                mesh = result[0]
                print("[Hunyuan3D2mvGenerator] Mesh type: %s" % type(mesh))
        finally:
            stop_evt.set()
            if progress_thread:
                progress_thread.join(timeout=1.0)

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 94, "Validating and exporting mesh...")
        
        # Validate mesh before export
        if mesh is None:
            raise RuntimeError("Generated mesh is None")
        
        if not hasattr(mesh, 'vertices') or mesh.vertices is None or len(mesh.vertices) == 0:
            raise RuntimeError("Generated mesh has no vertices")
        
        if not hasattr(mesh, 'faces') or mesh.faces is None or len(mesh.faces) == 0:
            raise RuntimeError("Generated mesh has no faces")
        
        print("[Hunyuan3D2mvGenerator] Mesh validation passed: %d vertices, %d faces" % (len(mesh.vertices), len(mesh.faces)))
        
        # Check mesh bounds
        bounds = mesh.bounds
        print("[Hunyuan3D2mvGenerator] Mesh bounds: %s" % bounds)
        
        extents = mesh.extents
        print("[Hunyuan3D2mvGenerator] Mesh extents: %s" % extents)
        
        if extents is not None and all(e > 0 for e in extents):
            print("[Hunyuan3D2mvGenerator] Mesh has valid extents")
        else:
            print("[Hunyuan3D2mvGenerator] WARNING: Mesh may have degenerate geometry (zero extent in some axes)")
        
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / ("%d_%s.glb" % (int(time.time()), uuid.uuid4().hex[:8]))
        mesh.export(str(out_path))
        print("[Hunyuan3D2mvGenerator] Exported GLB to: %s" % out_path)

        self._report(progress_cb, 100, "Done")
        return str(out_path)

    def _optional_view_image(self, params, view_name, remove_bg):
        path_key = "%s_image_path" % view_name
        data_key = "%s_image" % view_name

        path = params.get(path_key)
        if isinstance(path, str) and path.strip() and os.path.isfile(path):
            return self._preprocess_path(path, remove_bg=remove_bg)

        raw = params.get(data_key)
        if raw in (None, ""):
            return None

        if isinstance(raw, str):
            if params.get(data_key + "_is_b64"):
                raw = base64.b64decode(_strip_data_url(raw))
            elif os.path.isfile(raw):
                return self._preprocess_path(raw, remove_bg=remove_bg)
            else:
                try:
                    raw = base64.b64decode(_strip_data_url(raw), validate=True)
                except Exception:
                    print("[Hunyuan3D2mvGenerator] Ignoring %s: not a file or base64 image." % data_key)
                    return None

        if isinstance(raw, bytearray):
            raw = bytes(raw)
        if not isinstance(raw, bytes):
            print("[Hunyuan3D2mvGenerator] Ignoring %s: unsupported value type %s." % (data_key, type(raw).__name__))
            return None

        return self._preprocess_bytes(raw, remove_bg=remove_bg)

    def _preprocess_bytes(self, image_bytes, remove_bg=True):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._remove_bg(img) if remove_bg else img

    def _preprocess_path(self, path, remove_bg=True):
        img = Image.open(path).convert("RGB")
        return self._remove_bg(img) if remove_bg else img

    def _remove_bg(self, img):
        try:
            return self._rembg(img)
        except Exception as exc:
            print("[Hunyuan3D2mvGenerator] Background removal failed, using original image: %s" % exc)
            return img

    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download

        repo_id = self.hf_repo or _HF_REPO_ID
        manifest_skips = list(getattr(self, "hf_skip_prefixes", []) or [])
        ignore = []
        for pattern in manifest_skips:
            ignore.append(pattern)
            if isinstance(pattern, str) and pattern.endswith("/"):
                ignore.append(pattern + "*")
        ignore += [
            "*.md",
            "*.txt",
            "LICENSE",
            "NOTICE",
            "Notice.txt",
            ".gitattributes",
        ]
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3D2mvGenerator] Downloading weights from %s ..." % repo_id)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(self.model_dir),
            ignore_patterns=ignore,
        )
        print("[Hunyuan3D2mvGenerator] Weights downloaded.")


class Hunyuan3D2mvTurboGenerator(Hunyuan3D2mvGenerator):
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv-turbo"


class Hunyuan3D2mvFastGenerator(Hunyuan3D2mvGenerator):
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv-fast"


class Hunyuan3D2mvStandardGenerator(Hunyuan3D2mvGenerator):
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv"
