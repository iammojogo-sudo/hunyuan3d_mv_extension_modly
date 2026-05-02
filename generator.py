"""
Hunyuan3D-2mv - Modly extension generator.

Pipeline:
  1. Preprocess the uploaded front image and any optional side views.
  2. Run Hunyuan3DDiTFlowMatchingPipeline with front/left/back/right inputs.
  3. Export an untextured GLB mesh to the Modly workspace.

Texture node (separate):
  1. Load the untextured GLB.
  2. Preprocess front image (and optional side views) for the paint pipeline.
  3. Run Hunyuan3DPaintPipeline to bake UV texture.
  4. Export a textured GLB.
"""
import base64
import io
import os
import sys
import tempfile
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


_HF_REPO_ID    = "tencent/Hunyuan3D-2mv"
_PAINT_REPO_ID = "tencent/Hunyuan3D-2"

_SUBFOLDERS = {
    "hunyuan3d-dit-v2-mv-turbo": "hunyuan3d-dit-v2-mv-turbo",
    "hunyuan3d-dit-v2-mv-fast":  "hunyuan3d-dit-v2-mv-fast",
    "hunyuan3d-dit-v2-mv":       "hunyuan3d-dit-v2-mv",
}

_PAINT_SUBFOLDERS = {
    "hunyuan3d-paint-v2-0-turbo": "hunyuan3d-paint-v2-0-turbo",
    "hunyuan3d-paint-v2-0":       "hunyuan3d-paint-v2-0",
}

_DELIGHT_SUBFOLDER = "hunyuan3d-delight-v2-0"

# GLB files always start with these 4 magic bytes: "glTF"
_GLB_MAGIC = b"glTF"


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
    MODEL_ID      = "hunyuan3d2mv"
    DISPLAY_NAME  = "Hunyuan3D-2mv"
    VRAM_GB       = 8
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv-turbo"

    # ------------------------------------------------------------------
    # Download checks
    # ------------------------------------------------------------------

    def is_downloaded(self) -> bool:
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        marker = self.model_dir / self.MODEL_VARIANT / "model.fp16.safetensors"
        return marker.exists()

    def _is_paint_downloaded(self, paint_variant):
        delight_path = self.model_dir / _DELIGHT_SUBFOLDER
        paint_path   = self.model_dir / _PAINT_SUBFOLDERS[paint_variant]
        return delight_path.exists() and paint_path.exists()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_hy3dgen_on_path(self):
        repo_dir = Path(__file__).parent / "Hunyuan3D-2"
        if not repo_dir.exists():
            raise RuntimeError(
                "Hunyuan3D-2 source not found at %s. Please reinstall or repair the extension."
                % repo_dir
            )
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self):
        if self._model is not None:
            return

        if not self.is_downloaded():
            self._download_weights()

        self._ensure_hy3dgen_on_path()

        import torch
        from hy3dgen.rembg import BackgroundRemover
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        self._device               = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype                = torch.float16 if self._device == "cuda" else torch.float32
        self._rembg                = BackgroundRemover()
        self._loaded_variant       = None
        self._pipeline             = None
        self._loaded_paint_variant = None
        self._paint_pipeline       = None
        self._Pipeline             = Hunyuan3DDiTFlowMatchingPipeline
        self._model                = True
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
            local_files_only=True,
        )
        self._loaded_variant = variant
        print("[Hunyuan3D2mvGenerator] Variant loaded: %s" % variant)

    def _ensure_custom_rasterizer_importable(self):
        """
        Make sure `import custom_rasterizer_kernel` will succeed (this is what
        render.py inside the custom_rasterizer package actually imports).

        Resolution order:
          1. Already importable — done.
          2. Pre-built .pyd/.so exists in the source dir — add to sys.path.
          3. Neither — JIT-compile with torch.utils.cpp_extension.load so the
             user never has to touch a terminal. PyTorch caches the result in
             %TORCH_EXTENSIONS_DIR% so subsequent runs are instant.
        """
        # Step 1: already importable (site-packages install worked)
        try:
            import custom_rasterizer_kernel  # noqa: F401
            return
        except ModuleNotFoundError:
            pass

        rast_dir = Path(__file__).parent / "Hunyuan3D-2" / "hy3dgen" / "texgen" / "custom_rasterizer"
        if not rast_dir.exists():
            raise RuntimeError(
                "custom_rasterizer source directory not found at %s.\n"
                "Please reinstall or repair the extension." % rast_dir
            )

        kernel_dir = rast_dir / "lib" / "custom_rasterizer_kernel"

        # Step 2: pre-built artifact sitting in the source tree
        built = (
            list(rast_dir.glob("custom_rasterizer_kernel*.pyd")) +
            list(rast_dir.glob("custom_rasterizer_kernel*.so")) +
            list(kernel_dir.glob("custom_rasterizer_kernel*.pyd")) +
            list(kernel_dir.glob("custom_rasterizer_kernel*.so"))
        )
        if built:
            for search_dir in (str(rast_dir), str(kernel_dir)):
                if search_dir not in sys.path:
                    sys.path.insert(0, search_dir)
            try:
                import custom_rasterizer_kernel  # noqa: F401
                print("[Hunyuan3D2mvGenerator] custom_rasterizer_kernel import OK (pre-built).")
                return
            except ModuleNotFoundError:
                pass  # fall through to JIT

        # Step 3: JIT compile — happens once, cached by PyTorch automatically
        print("[Hunyuan3D2mvGenerator] custom_rasterizer_kernel not found — JIT compiling (this takes a minute on first run)...")
        try:
            import torch.utils.cpp_extension as cpp_ext

            # Ensure the venv's Scripts/bin dir is on PATH so ninja is findable
            venv_dir = Path(__file__).parent / "venv"
            scripts_dir = venv_dir / ("Scripts" if os.name == "nt" else "bin")
            if scripts_dir.exists():
                current_path = os.environ.get("PATH", "")
                if str(scripts_dir) not in current_path:
                    os.environ["PATH"] = str(scripts_dir) + os.pathsep + current_path

            # --- Robust CUDA_HOME resolution ---
            # cpp_extension.CUDA_HOME is a module-level var frozen at import time,
            # so we must patch it directly in addition to os.environ.
            import shutil as _shutil

            def _resolve_cuda_home():
                # 1. Already in env
                for k in ("CUDA_HOME", "CUDA_PATH"):
                    v = os.environ.get(k)
                    if v and Path(v).exists():
                        return v
                # 2. NVIDIA versioned env vars e.g. CUDA_PATH_V12_4
                for k, v in os.environ.items():
                    if k.startswith("CUDA_PATH_V") and v and Path(v).exists():
                        return v
                # 3. Derive from nvcc on PATH
                nvcc = _shutil.which("nvcc") or _shutil.which("nvcc.exe")
                if nvcc:
                    cuda_root = str(Path(nvcc).parent.parent)
                    if Path(cuda_root).exists():
                        return cuda_root
                # 4. Windows registry
                if os.name == "nt":
                    try:
                        import winreg
                        reg_path = r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA"
                        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
                        n = winreg.QueryInfoKey(key)[0]
                        versions = []
                        for i in range(n):
                            sub = winreg.EnumKey(key, i)
                            try:
                                sk = winreg.OpenKey(key, sub)
                                p, _ = winreg.QueryValueEx(sk, "InstallDir")
                                if p and Path(p).exists():
                                    versions.append(p)
                            except OSError:
                                pass
                        if versions:
                            return versions[-1]
                    except OSError:
                        pass
                # 5. Filesystem scan
                cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
                if cuda_base.exists():
                    dirs = sorted([d for d in cuda_base.iterdir() if d.is_dir()], reverse=True)
                    if dirs:
                        return str(dirs[0])
                return None

            cuda_home_val = _resolve_cuda_home()
            if cuda_home_val:
                os.environ["CUDA_HOME"] = cuda_home_val
                cpp_ext.CUDA_HOME = cuda_home_val
                print("[Hunyuan3D2mvGenerator] Resolved CUDA_HOME: %s" % cuda_home_val)
            else:
                print("[Hunyuan3D2mvGenerator] WARNING: Could not resolve CUDA_HOME automatically")

            # Auto-detect cl.exe (MSVC) if not already on PATH
            if os.name == "nt":
                import shutil as _shutil
                if not _shutil.which("cl"):
                    for vs_base in [
                        r"C:\Program Files\Microsoft Visual Studio",
                        r"C:\Program Files (x86)\Microsoft Visual Studio",
                    ]:
                        vs_p = Path(vs_base)
                        if vs_p.exists():
                            cl_hits = [
                                p for p in vs_p.rglob("cl.exe")
                                if "x64" in str(p) or "amd64" in str(p).lower()
                            ]
                            if cl_hits:
                                cl_dir = str(cl_hits[0].parent)
                                os.environ["PATH"] = cl_dir + os.pathsep + os.environ.get("PATH", "")
                                print("[Hunyuan3D2mvGenerator] Auto-found cl.exe: %s" % cl_hits[0])
                                break

            cpp_sources = [
                str(kernel_dir / "rasterizer.cpp"),
                str(kernel_dir / "grid_neighbor.cpp"),
            ]
            cuda_sources = [
                str(kernel_dir / "rasterizer_gpu.cu"),
            ]

            cpp_ext.load(
                name="custom_rasterizer_kernel",
                sources=cpp_sources + cuda_sources,
                extra_include_paths=[str(kernel_dir)],
                verbose=True,
            )

            import custom_rasterizer_kernel  # noqa: F401
            print("[Hunyuan3D2mvGenerator] custom_rasterizer_kernel JIT compile OK.")
        except Exception as exc:
            raise RuntimeError(
                "custom_rasterizer_kernel could not be compiled automatically.\n\n"
                "Error: %s\n\n"
                "Make sure the CUDA toolkit is installed and matches your PyTorch build.\n"
                "On Windows, Visual Studio C++ build tools are also required." % exc
            ) from exc

    def _load_paint_pipeline(self, paint_variant):
        paint_variant = paint_variant if paint_variant in _PAINT_SUBFOLDERS else "hunyuan3d-paint-v2-0-turbo"
        if self._loaded_paint_variant == paint_variant:
            return

        import torch

        print("[Hunyuan3D2mvGenerator] Loading paint variant: %s ..." % paint_variant)
        if self._paint_pipeline is not None:
            del self._paint_pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not self._is_paint_downloaded(paint_variant):
            self._download_paint_weights(paint_variant)

        # Ensure the compiled C rasterizer is importable BEFORE loading the pipeline.
        # MeshRender imports custom_rasterizer at __init__ time; if it is not on
        # sys.path the whole pipeline load crashes with ModuleNotFoundError.
        self._ensure_custom_rasterizer_importable()

        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        self._paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            str(self.model_dir),
            subfolder=_PAINT_SUBFOLDERS[paint_variant],
        )
        self._loaded_paint_variant = paint_variant
        print("[Hunyuan3D2mvGenerator] Paint variant loaded: %s" % paint_variant)

    def unload(self):
        self._pipeline             = None
        self._loaded_variant       = None
        self._paint_pipeline       = None
        self._loaded_paint_variant = None
        self._model                = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # generate() — entry point for BOTH nodes via runner.py
    #
    # Modly passes all node inputs through generate(image_bytes, params).
    # When the texture node runs, it passes GLB bytes instead of image
    # bytes. We detect this via the GLB magic bytes and route accordingly.
    # ------------------------------------------------------------------

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        # ---- Route to texture if input is a GLB mesh -----------------
        if isinstance(image_bytes, (bytes, bytearray)) and image_bytes[:4] == _GLB_MAGIC:
            print("[Hunyuan3D2mvGenerator] GLB input detected — routing to texture pipeline.")
            return self._generate_texture_from_bytes(image_bytes, params, progress_cb, cancel_event)

        # ---- Otherwise run shape generation --------------------------
        return self._generate_shape(image_bytes, params, progress_cb, cancel_event)

    # ------------------------------------------------------------------
    # Shape generation
    # ------------------------------------------------------------------

    def _generate_shape(self, image_bytes, params, progress_cb=None, cancel_event=None):
        import torch

        params         = params or {}
        variant        = params.get("model_variant") or self.MODEL_VARIANT
        steps          = _safe_int(params.get("num_inference_steps"), 30)
        octree_res     = _safe_int(params.get("octree_resolution"), 380)
        seed           = _safe_int(params.get("seed"), 42)
        guidance_scale = _safe_float(params.get("guidance_scale"), 5.0)
        num_chunks     = _safe_int(params.get("num_chunks"), 8000)
        box_v          = _safe_float(params.get("box_v"), 1.01)
        mc_level       = _safe_float(params.get("mc_level"), 0.0)
        remove_bg      = _safe_bool(params.get("remove_bg"), True)

        print(
            "[Hunyuan3D2mvGenerator] Shape params: variant=%s steps=%s octree=%s "
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
        stop_evt        = threading.Event()
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
                mesh = result[0]
        finally:
            stop_evt.set()
            if progress_thread:
                progress_thread.join(timeout=1.0)

        self._check_cancelled(cancel_event)

        self._report(progress_cb, 94, "Validating mesh...")

        if mesh is None:
            raise RuntimeError("Generated mesh is None")
        if not hasattr(mesh, "vertices") or mesh.vertices is None or len(mesh.vertices) == 0:
            raise RuntimeError("Generated mesh has no vertices")
        if not hasattr(mesh, "faces") or mesh.faces is None or len(mesh.faces) == 0:
            raise RuntimeError("Generated mesh has no faces")

        print("[Hunyuan3D2mvGenerator] Mesh validated: %d vertices, %d faces"
              % (len(mesh.vertices), len(mesh.faces)))

        self._report(progress_cb, 98, "Exporting mesh...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / ("%d_%s.glb" % (int(time.time()), uuid.uuid4().hex[:8]))
        mesh.export(str(out_path))
        print("[Hunyuan3D2mvGenerator] Exported GLB to: %s" % out_path)

        self._report(progress_cb, 100, "Done")
        return str(out_path)

    # ------------------------------------------------------------------
    # Texture routing — called when generate() receives GLB bytes
    # ------------------------------------------------------------------

    def _generate_texture_from_bytes(self, glb_bytes, params, progress_cb=None, cancel_event=None):
        """
        Saves the incoming GLB bytes to a temp file then calls texture().
        This is the path taken when the texture node runs through runner.py.
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                f.write(glb_bytes)
                tmp_path = f.name
            print("[Hunyuan3D2mvGenerator] Saved GLB to temp: %s" % tmp_path)
            return self.texture(tmp_path, params, progress_cb, cancel_event)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Texture node — full pipeline
    # ------------------------------------------------------------------

    def texture(self, mesh_path, params, progress_cb=None, cancel_event=None):
        """
        Apply texture to an existing untextured GLB mesh.

        Args:
            mesh_path: Path to the input .glb file.
            params:    Dict of texture parameters from the manifest schema.

        Returns:
            Path to the textured .glb file.
        """
        import trimesh

        params        = params or {}
        paint_variant = params.get("texture_variant") or "hunyuan3d-paint-v2-0-turbo"
        remove_bg     = _safe_bool(params.get("remove_bg"), True)

        print("[Hunyuan3D2mvGenerator] Texture params: variant=%s remove_bg=%s"
              % (paint_variant, remove_bg))

        # ---- Load mesh -----------------------------------------------
        self._report(progress_cb, 5, "Loading mesh...")
        mesh = trimesh.load(str(mesh_path), force="mesh")
        if mesh is None or len(mesh.vertices) == 0:
            raise RuntimeError("Could not load mesh from: %s" % mesh_path)
        print("[Hunyuan3D2mvGenerator] Mesh loaded: %d vertices, %d faces"
              % (len(mesh.vertices), len(mesh.faces)))
        self._check_cancelled(cancel_event)

        # ---- Load reference images -----------------------------------
        self._report(progress_cb, 8, "Preprocessing reference images...")

        front_path = params.get("front_image_path", "").strip()
        if not front_path or not os.path.isfile(front_path):
            raise RuntimeError(
                "front_image_path is required for texture generation. "
                "Please supply the same front view used for shape generation."
            )

        images = [self._preprocess_path(front_path, remove_bg=remove_bg)]

        for view_name, pct in (("left", 12), ("back", 15), ("right", 18)):
            img = self._optional_view_image(params, view_name, remove_bg)
            if img is None:
                continue
            self._report(progress_cb, pct, "Preprocessing %s view..." % view_name)
            images.append(img)
            self._check_cancelled(cancel_event)

        print("[Hunyuan3D2mvGenerator] Using %d reference image(s) for texturing." % len(images))

        # ---- Load paint pipeline -------------------------------------
        self._report(progress_cb, 20, "Loading texture pipeline...")
        self._load_paint_pipeline(paint_variant)
        self._check_cancelled(cancel_event)

        # ---- Run texture pipeline ------------------------------------
        self._report(progress_cb, 25, "Applying texture...")
        stop_evt   = threading.Event()
        tex_thread = None
        if progress_cb:
            tex_thread = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 25, 95, "Applying texture...", stop_evt),
                daemon=True,
            )
            tex_thread.start()

        try:
            textured_mesh = self._paint_pipeline(mesh, images)
        finally:
            stop_evt.set()
            if tex_thread:
                tex_thread.join(timeout=1.0)

        self._check_cancelled(cancel_event)

        if textured_mesh is None:
            raise RuntimeError("Texture pipeline returned None.")

        print("[Hunyuan3D2mvGenerator] Texture applied successfully.")

        # ---- Export --------------------------------------------------
        self._report(progress_cb, 98, "Exporting textured mesh...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / ("%d_%s_textured.glb" % (int(time.time()), uuid.uuid4().hex[:8]))
        textured_mesh.export(str(out_path))
        print("[Hunyuan3D2mvGenerator] Exported textured GLB to: %s" % out_path)

        self._report(progress_cb, 100, "Done")
        return str(out_path)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

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
            print("[Hunyuan3D2mvGenerator] Ignoring %s: unsupported value type %s."
                  % (data_key, type(raw).__name__))
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
            print("[Hunyuan3D2mvGenerator] Background removal failed, using original: %s" % exc)
            return img

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download

        repo_id        = self.hf_repo or _HF_REPO_ID
        manifest_skips = list(getattr(self, "hf_skip_prefixes", []) or [])
        ignore = []
        for pattern in manifest_skips:
            ignore.append(pattern)
            if isinstance(pattern, str) and pattern.endswith("/"):
                ignore.append(pattern + "*")
        ignore += ["*.md", "*.txt", "LICENSE", "NOTICE", "Notice.txt", ".gitattributes"]

        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[Hunyuan3D2mvGenerator] Downloading shape weights from %s ..." % repo_id)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(self.model_dir),
            ignore_patterns=ignore,
        )
        print("[Hunyuan3D2mvGenerator] Shape weights downloaded.")

    def _download_paint_weights(self, paint_variant):
        from huggingface_hub import snapshot_download

        paint_variant = paint_variant if paint_variant in _PAINT_SUBFOLDERS else "hunyuan3d-paint-v2-0-turbo"
        allow = [
            "%s/*" % _DELIGHT_SUBFOLDER,
            "%s/*" % _PAINT_SUBFOLDERS[paint_variant],
        ]
        print("[Hunyuan3D2mvGenerator] Downloading paint weights from %s ..." % _PAINT_REPO_ID)
        snapshot_download(
            repo_id=_PAINT_REPO_ID,
            local_dir=str(self.model_dir),
            allow_patterns=allow,
        )
        print("[Hunyuan3D2mvGenerator] Paint weights downloaded.")


# ---------------------------------------------------------------------------
# Variant subclasses
# ---------------------------------------------------------------------------

class Hunyuan3D2mvTurboGenerator(Hunyuan3D2mvGenerator):
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv-turbo"


class Hunyuan3D2mvFastGenerator(Hunyuan3D2mvGenerator):
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv-fast"


class Hunyuan3D2mvStandardGenerator(Hunyuan3D2mvGenerator):
    MODEL_VARIANT = "hunyuan3d-dit-v2-mv"