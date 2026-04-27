"""
Hunyuan3D-2mv - Modly extension generator.

Single generator class handles both nodes:
  - generate-turbo / generate-fast / generate  -> mesh generation
  - texture-turbo                               -> texture generation

Routing is based on self.model_dir.name which matches the node id
set by Modly via the MODEL_DIR env var.
"""
import base64
import io
import json
import os
import random
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled


_TEXTURE_NODE_IDS = {"texture-turbo", "texture-fast", "texture"}

_PAINT_SUBFOLDERS = {
    "texture-turbo": "hunyuan3d-paint-v2-0-turbo",
    "texture-fast":  "hunyuan3d-paint-v2-0-fast",
    "texture":       "hunyuan3d-paint-v2-0",
}
_DELIGHT_SUBFOLDER = "hunyuan3d-delight-v2-0"

# Shape node subfolder mapping: node_id -> weight subfolder name
_SHAPE_SUBFOLDERS = {
    "generate-turbo": "hunyuan3d-dit-v2-mv-turbo",
    "generate-fast":  "hunyuan3d-dit-v2-mv-fast",
    "generate":       "hunyuan3d-dit-v2-mv",
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


def _safe_bool(val, default):
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return default


# ------------------------------------------------------------------ #
# smart_load_model patch (shape pipeline only)
# ------------------------------------------------------------------ #

def _patch_smart_load_model():
    """
    Patch hy3dgen's smart_load_model so it works with a pre-downloaded local
    model directory even when the internal subfolder it expects doesn't exist.
    """
    try:
        import hy3dgen.shapegen.utils as _utils
        _original = _utils.smart_load_model

        def _patched(model_id, subfolder=None, **kwargs):
            base = Path(str(model_id))

            if subfolder:
                sub = base / subfolder
                if sub.exists() and sub.is_dir():
                    return _original(model_id, subfolder=subfolder, **kwargs)

            if base.exists() and base.is_dir():
                yaml_files = sorted(base.glob("*.yaml"))
                safetensor_files = sorted(base.glob("*.safetensors"))
                ckpt_files = sorted(base.glob("*.ckpt"))
                bin_files = sorted(base.glob("*.bin"))
                weight_files = safetensor_files or ckpt_files or bin_files

                if yaml_files and weight_files:
                    config_path = str(yaml_files[0])
                    ckpt_path = str(weight_files[0])
                    print(
                        "[smart_load_model patched] Using local model directly:\n"
                        "  config : %s\n"
                        "  weights: %s" % (config_path, ckpt_path)
                    )
                    return config_path, ckpt_path

                if weight_files:
                    ckpt_path = str(weight_files[0])
                    print(
                        "[smart_load_model patched] Using local weights (no yaml):\n"
                        "  weights: %s" % ckpt_path
                    )
                    return str(base), ckpt_path

            return _original(model_id, subfolder=subfolder, **kwargs)

        _utils.smart_load_model = _patched

        import hy3dgen.shapegen.pipelines as _pipelines
        if hasattr(_pipelines, "smart_load_model"):
            _pipelines.smart_load_model = _patched

        print("[Hunyuan3D2mvGenerator] smart_load_model patched in utils + pipelines.")
    except Exception as e:
        print("[Hunyuan3D2mvGenerator] Warning: could not patch smart_load_model: %s" % e)


# ------------------------------------------------------------------ #
# nvdiffrast MeshRender patch (texture pipeline)
# ------------------------------------------------------------------ #

def _ensure_nvdiffrast() -> bool:
    """Return True if nvdiffrast is importable; auto-install it if not."""
    try:
        import nvdiffrast  # noqa
        return True
    except ImportError:
        pass
    import subprocess as _sp
    print("[Hunyuan3D2mvGenerator] nvdiffrast not found — auto-installing...")
    try:
        _sp.run([sys.executable, "-m", "pip", "install", "nvdiffrast"], check=True)
        print("[Hunyuan3D2mvGenerator] nvdiffrast installed successfully.")
        return True
    except Exception as err:
        print(
            "[Hunyuan3D2mvGenerator] nvdiffrast auto-install failed: %s\n"
            "  Run manually: %s -m pip install nvdiffrast" % (err, sys.executable)
        )
        return False


def _patch_mesh_render():
    """
    Monkey-patch MeshRender to use nvdiffrast instead of custom_rasterizer.

    nvdiffrast ships as a prebuilt pip wheel — no CUDA toolkit needed on the
    user machine.  We patch five methods on the MeshRender *class object* so
    every future (and already-imported) instance uses the new backend.

    Patched methods
    ---------------
    __init__           creates an nvdiffrast GL/CUDA context
    raster_rasterize   dr.rasterize()
    raster_interpolate dr.interpolate()
    raster_texture     dr.texture()
    raster_antialias   dr.antialias()
    """
    # Step 1 — make sure nvdiffrast is actually installed
    if not _ensure_nvdiffrast():
        print("[Hunyuan3D2mvGenerator] Cannot patch MeshRender: nvdiffrast unavailable.")
        return

    # Step 2 — imports (safe now that nvdiffrast is confirmed present)
    try:
        import nvdiffrast.torch as dr
        import torch
        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
        from hy3dgen.texgen.differentiable_renderer.camera_utils import (
            get_orthographic_projection_matrix,
            get_perspective_projection_matrix,
        )
    except ImportError as e:
        print("[Hunyuan3D2mvGenerator] _patch_mesh_render: import failed — %s" % e)
        return

    # Step 3 — also purge the module from sys.modules so a fresh import in
    # pipelines.py picks up our patched class (handles the edge case where
    # the module was already loaded before this function ran).
    import sys as _sys
    for mod_key in list(_sys.modules.keys()):
        if "mesh_render" in mod_key or "texgen" in mod_key:
            del _sys.modules[mod_key]

    # Re-import after purge so our patch target is the definitive class object.
    try:
        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender  # noqa: F811
    except ImportError as e:
        print("[Hunyuan3D2mvGenerator] _patch_mesh_render: re-import failed — %s" % e)
        return

    # --------------------------------------------------------------------- #
    # Replacement implementations
    # --------------------------------------------------------------------- #

    def _nv_init(
        self,
        camera_distance=1.45,
        camera_type="orth",
        default_resolution=1024,
        texture_size=1024,
        use_antialias=True,
        max_mip_level=None,
        filter_mode="linear",
        bake_mode="linear",
        raster_mode="nvdiffrast",
        device="cuda",
    ):
        self.device = device
        self.set_default_render_resolution(default_resolution)
        self.set_default_texture_resolution(texture_size)
        self.camera_distance = camera_distance
        self.use_antialias = use_antialias
        self.max_mip_level = max_mip_level
        self.filter_mode = filter_mode
        self.bake_angle_thres = 75
        self.bake_unreliable_kernel_size = int(
            (2 / 512) * max(self.default_resolution[0], self.default_resolution[1])
        )
        self.bake_mode = bake_mode
        self.raster_mode = "nvdiffrast"

        # Prefer the CUDA context (no display server required).
        try:
            self._glctx = dr.RasterizeCudaContext()
            print("[MeshRender/nvdiffrast] CUDA rasteriser context created.")
        except Exception as cuda_err:
            print(
                "[MeshRender/nvdiffrast] CUDA context failed (%s), "
                "trying OpenGL context..." % cuda_err
            )
            self._glctx = dr.RasterizeGLContext()
            print("[MeshRender/nvdiffrast] OpenGL rasteriser context created.")

        if camera_type == "orth":
            self.ortho_scale = 1.2
            self.camera_proj_mat = get_orthographic_projection_matrix(
                left=-self.ortho_scale * 0.5,
                right=self.ortho_scale * 0.5,
                bottom=-self.ortho_scale * 0.5,
                top=self.ortho_scale * 0.5,
                near=0.1,
                far=100,
            )
        elif camera_type == "perspective":
            self.camera_proj_mat = get_perspective_projection_matrix(
                49.13,
                self.default_resolution[1] / self.default_resolution[0],
                0.01,
                100.0,
            )
        else:
            raise ValueError("Unknown camera_type: %s" % camera_type)

    def _nv_raster_rasterize(self, pos, tri, resolution, ranges=None, grad_db=True):
        # nvdiffrast output: [B, H, W, 4]
        #   channel 3 = triangle_id (1-indexed; 0 = background)
        # The existing visible_mask code does torch.clamp(rast[..., -1:], 0, 1)
        # which maps 0->0 (bg) and any tri_id->1 (visible). Works without change.
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        pos = pos.float().contiguous()
        tri = tri.to(torch.int32).contiguous()
        rast_out, rast_out_db = dr.rasterize(self._glctx, pos, tri, resolution)
        return rast_out, rast_out_db

    def _nv_raster_interpolate(
        self, attr, rast_out, tri, rast_db=None, diff_attrs=None
    ):
        if attr.dim() == 2:
            attr = attr.unsqueeze(0)
        attr = attr.float().contiguous()
        tri = tri.to(torch.int32).contiguous()
        if diff_attrs is not None and rast_db is not None:
            texc, texd = dr.interpolate(
                attr, rast_out, tri, rast_db=rast_db, diff_attrs=diff_attrs
            )
        else:
            texc, texd = dr.interpolate(attr, rast_out, tri)
        return texc, texd

    def _nv_raster_texture(
        self,
        tex,
        uv,
        uv_da=None,
        mip_level_bias=None,
        mip=None,
        filter_mode="auto",
        boundary_mode="wrap",
        max_mip_level=None,
    ):
        return dr.texture(
            tex, uv,
            uv_da=uv_da,
            filter_mode=filter_mode,
            boundary_mode=boundary_mode,
            max_mip_level=max_mip_level,
        )

    def _nv_raster_antialias(
        self, color, rast, pos, tri, topology_hash=None, pos_gradient_boost=1.0
    ):
        if pos.dim() == 2:
            pos = pos.unsqueeze(0)
        pos = pos.float().contiguous()
        tri = tri.to(torch.int32).contiguous()
        return dr.antialias(color, rast, pos, tri)

    # Apply all five patches to the class object.
    MeshRender.__init__           = _nv_init
    MeshRender.raster_rasterize   = _nv_raster_rasterize
    MeshRender.raster_interpolate = _nv_raster_interpolate
    MeshRender.raster_texture     = _nv_raster_texture
    MeshRender.raster_antialias   = _nv_raster_antialias

    print("[Hunyuan3D2mvGenerator] MeshRender patched successfully → nvdiffrast.")


# ------------------------------------------------------------------ #
# Unified generator
# ------------------------------------------------------------------ #

class Hunyuan3D2mvGenerator(BaseGenerator):
    MODEL_ID = "hunyuan3d2mv"
    DISPLAY_NAME = "Hunyuan3D 2 Multiview"
    VRAM_GB = 8

    def __init__(self, model_dir, outputs_dir=None):
        super().__init__(model_dir, outputs_dir)
        # Second model slot for texture pipeline (shape uses self._model)
        self._texture_pipeline = None

    @property
    def _is_texture_node(self) -> bool:
        return self.model_dir.name in _TEXTURE_NODE_IDS

    # ------------------------------------------------------------------
    # params_schema — node-aware classmethod
    #
    # The runner calls GenClass.params_schema() BEFORE instantiation, so
    # we read MODEL_DIR from the environment to find the active node, then
    # load the manifest to return that node's params_schema.
    # ------------------------------------------------------------------

    @classmethod
    def params_schema(cls) -> list:
        try:
            # Determine the active node id from the MODEL_DIR env var.
            model_dir_env = os.environ.get("MODEL_DIR", "")
            node_id = Path(model_dir_env).name if model_dir_env else ""

            # Load the manifest sitting next to this generator file.
            ext_dir = Path(os.environ.get("EXTENSION_DIR", Path(__file__).parent))
            manifest_path = ext_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            nodes = manifest.get("nodes") or []
            if node_id:
                node = next((n for n in nodes if n.get("id") == node_id), None)
                if node:
                    schema = node.get("params_schema") or []
                    print(
                        "[Hunyuan3D2mvGenerator] params_schema: node=%s, %d params"
                        % (node_id, len(schema)),
                        file=sys.stderr,
                    )
                    return schema

            # Fallback: return first node's schema if node_id not matched.
            fallback = (nodes[0].get("params_schema") or []) if nodes else []
            print(
                "[Hunyuan3D2mvGenerator] params_schema: node_id=%r not matched, "
                "returning first node schema (%d params)" % (node_id, len(fallback)),
                file=sys.stderr,
            )
            return fallback

        except Exception as e:
            print(
                "[Hunyuan3D2mvGenerator] params_schema() failed: %s" % e,
                file=sys.stderr,
            )
            return []

    # ------------------------------------------------------------------
    # is_downloaded
    # ------------------------------------------------------------------

    def is_downloaded(self) -> bool:
        if self._is_texture_node:
            delight_path = self.model_dir / _DELIGHT_SUBFOLDER
            return delight_path.exists() and any(delight_path.glob("*.safetensors"))
        else:
            node_id = self.model_dir.name
            subfolder = _SHAPE_SUBFOLDERS.get(node_id, "hunyuan3d-dit-v2-mv-turbo")
            model_dir = self.model_dir / subfolder
            return model_dir.exists() and (model_dir / "model.fp16.safetensors").exists()

    # ------------------------------------------------------------------
    # load / unload
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._is_texture_node:
            self._load_texture()
        else:
            self._load_shape()

    def unload(self) -> None:
        super().unload()
        if self._texture_pipeline is not None:
            self._texture_pipeline = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Shape pipeline load
    # ------------------------------------------------------------------

    @staticmethod
    def _patch_pipeline_components(pipeline):
        """Inject a components property so enable_model_cpu_offload works."""
        def _components(self):
            return {
                "vae": self.vae,
                "model": self.model,
                "conditioner": self.conditioner,
                "image_processor": self.image_processor,
                "scheduler": self.scheduler,
            }

        klass = type(pipeline)
        patched_klass = type(klass.__name__ + "_patched", (klass,), {
            "components": property(_components),
        })
        pipeline.__class__ = patched_klass
        print("[Hunyuan3D2mvGenerator] components property injected onto pipeline.")

    def _load_shape(self) -> None:
        if self._model is not None:
            return

        print("[Hunyuan3D2mvGenerator] Loading shape pipeline...")

        import torch
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

        _patch_smart_load_model()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        node_id = self.model_dir.name
        subfolder = _SHAPE_SUBFOLDERS.get(node_id, "hunyuan3d-dit-v2-mv-turbo")
        model_path = self.model_dir / subfolder

        print("[Hunyuan3D2mvGenerator] Loading pipeline from %s" % model_path)

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(model_path),
            use_safetensors=True,
        )
        self._patch_pipeline_components(pipeline)

        vram_gb = 0.0
        if device == "cuda":
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                pass

        if device == "cuda" and vram_gb >= 10:
            print("[Hunyuan3D2mvGenerator] %.1f GB VRAM, enabling cpu offload." % vram_gb)
            try:
                pipeline.enable_model_cpu_offload(device=device)
            except Exception as e:
                print("[Hunyuan3D2mvGenerator] cpu offload failed (%s), using fp16 on GPU." % e)
                pipeline.to(device, torch.float16)
        else:
            print("[Hunyuan3D2mvGenerator] %.1f GB VRAM, keeping on GPU in fp16." % vram_gb)
            pipeline.to(device, torch.float16)

        self._model = pipeline
        print("[Hunyuan3D2mvGenerator] Shape pipeline loaded on %s." % device)

    # ------------------------------------------------------------------
    # Texture pipeline load
    # ------------------------------------------------------------------

    def _load_texture(self) -> None:
        if self._texture_pipeline is not None:
            return

        node_id = self.model_dir.name
        paint_subfolder = _PAINT_SUBFOLDERS.get(node_id, "hunyuan3d-paint-v2-0-turbo")
        paint_path = self.model_dir / paint_subfolder
        delight_path = self.model_dir / _DELIGHT_SUBFOLDER

        missing = [str(p) for p in (paint_path, delight_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "[Hunyuan3D2mvGenerator] Missing texture model folders:\n  %s\n"
                "Make sure the texture node has been fully downloaded in Modly."
                % "\n  ".join(missing)
            )

        print(
            "[Hunyuan3D2mvGenerator] Loading texture pipeline from %s\n"
            "  paint  : %s\n"
            "  delight: %s" % (self.model_dir, paint_path, delight_path)
        )

        # Patch MeshRender to use nvdiffrast before the pipeline class is
        # imported and instantiated. Also auto-installs nvdiffrast if missing.
        _patch_mesh_render()

        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        # from_pretrained local branch: needs model_dir as root, paint subfolder as subfolder.
        self._texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            str(self.model_dir),
            subfolder=paint_subfolder,
        )

        try:
            self._texture_pipeline.enable_model_cpu_offload()
            print("[Hunyuan3D2mvGenerator] Texture cpu offload enabled.")
        except Exception as e:
            print("[Hunyuan3D2mvGenerator] Texture cpu offload warning: %s" % e)

        print("[Hunyuan3D2mvGenerator] Texture pipeline ready.")

    # ------------------------------------------------------------------
    # Image preprocessing (shared)
    # ------------------------------------------------------------------

    def _preprocess(self, image_bytes: bytes, remove_bg: bool = True):
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode == "RGBA":
            remove_bg = False

        if remove_bg:
            try:
                from hy3dgen.rembg import BackgroundRemover
                remover = BackgroundRemover()
                img = remover(img)
                print("[Hunyuan3D2mvGenerator] Background removed")
            except Exception as e:
                print("[Hunyuan3D2mvGenerator] Background removal failed: %s" % e)

        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        return img

    def _optional_view_image(self, params: dict, view_name: str, remove_bg: bool = True):
        path_key = "%s_image_path" % view_name
        data_key = "%s_image" % view_name

        path = params.get(path_key)
        data = params.get(data_key)

        if path and os.path.exists(path):
            with open(path, "rb") as f:
                image_bytes = f.read()
            return self._preprocess(image_bytes, remove_bg=remove_bg)

        if data:
            if isinstance(data, str):
                try:
                    image_bytes = base64.b64decode(data)
                except Exception:
                    return None
            else:
                image_bytes = data
            return self._preprocess(image_bytes, remove_bg=remove_bg)

        return None

    # ------------------------------------------------------------------
    # generate — routes to shape or texture
    # ------------------------------------------------------------------

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        if self._is_texture_node:
            return self._generate_texture(image_bytes, params, progress_cb, cancel_event)
        else:
            return self._generate_shape(image_bytes, params, progress_cb, cancel_event)

    # ------------------------------------------------------------------
    # Shape generation
    # ------------------------------------------------------------------

    def _generate_shape(self, image_bytes, params, progress_cb, cancel_event):
        import torch

        params = params or {}
        node_id = self.model_dir.name
        default_variant = _SHAPE_SUBFOLDERS.get(node_id, "hunyuan3d-dit-v2-mv-turbo")
        variant = params.get("model_variant", default_variant)
        steps = _safe_int(params.get("num_inference_steps"), 30)
        octree_res = _safe_int(params.get("octree_resolution"), 380)
        seed = _safe_int(params.get("seed"), -1)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        guidance_scale = _safe_float(params.get("guidance_scale"), 5.0)
        num_chunks = _safe_int(params.get("num_chunks"), 8000)
        box_v = _safe_float(params.get("box_v"), 1.01)
        mc_level = _safe_float(params.get("mc_level"), 0.0)
        remove_bg = _safe_bool(params.get("remove_bg"), True)

        print(
            "[Hunyuan3D2mvGenerator] Shape params: variant=%s steps=%s octree=%s "
            "guidance=%.2f chunks=%s box_v=%.3f mc_level=%.4f remove_bg=%s seed=%s"
            % (variant, steps, octree_res, guidance_scale, num_chunks, box_v, mc_level, remove_bg, seed)
        )

        self._report(progress_cb, 5, "Preprocessing front view...")
        front_image = self._preprocess(image_bytes, remove_bg=remove_bg)
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

        image_dict = {"front": front_image}
        for view_name, pct in (("left", 10), ("back", 14), ("right", 18)):
            image = self._optional_view_image(params, view_name, remove_bg)
            if image is None:
                continue
            self._report(progress_cb, pct, "Preprocessing %s view..." % view_name)
            image_dict[view_name] = image
            if cancel_event and cancel_event.is_set():
                raise GenerationCancelled()

        print("[Hunyuan3D2mvGenerator] image_dict keys: %s" % list(image_dict.keys()))

        self._report(progress_cb, 22, "Loading model variant...")
        if self._model is None:
            self.load()
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

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
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
            with torch.no_grad():
                result = self._model(
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
                mesh = result[0] if isinstance(result, (list, tuple)) else result
                print("[Hunyuan3D2mvGenerator] Mesh type: %s" % type(mesh))
        finally:
            stop_evt.set()
            if progress_thread:
                progress_thread.join(timeout=1.0)

        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

        self._report(progress_cb, 94, "Validating and exporting mesh...")

        if mesh is None:
            raise RuntimeError("Generated mesh is None")
        if not hasattr(mesh, "vertices") or mesh.vertices is None or len(mesh.vertices) == 0:
            raise RuntimeError("Generated mesh has no vertices")
        if not hasattr(mesh, "faces") or mesh.faces is None or len(mesh.faces) == 0:
            raise RuntimeError("Generated mesh has no faces")

        print("[Hunyuan3D2mvGenerator] Mesh: %d vertices, %d faces" % (len(mesh.vertices), len(mesh.faces)))

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / ("%d_%s.glb" % (int(time.time()), uuid.uuid4().hex[:8]))
        mesh.export(str(out_path))
        print("[Hunyuan3D2mvGenerator] Exported GLB to: %s" % out_path)

        self._report(progress_cb, 100, "Done")
        return out_path

    # ------------------------------------------------------------------
    # Texture generation
    # ------------------------------------------------------------------

    def _generate_texture(self, image_bytes, params, progress_cb, cancel_event):
        import torch
        import trimesh

        params = params or {}
        remove_bg = _safe_bool(params.get("remove_bg"), True)

        print("[Hunyuan3D2mvGenerator] Texture params: remove_bg=%s" % remove_bg)

        # The runner passes the mesh GLB path via params["mesh_path"] or
        # params["input_path"] — Modly sets this when the node input is a mesh.
        mesh_path = params.get("mesh_path") or params.get("input_path")
        if not mesh_path:
            raise ValueError(
                "[Hunyuan3D2mvGenerator] Texture node requires a mesh input. "
                "No mesh_path or input_path found in params. Keys: %s" % list(params.keys())
            )

        self._report(progress_cb, 5, "Loading mesh...")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError("Mesh not found at %s" % mesh_path)
        mesh = trimesh.load(mesh_path)
        print("[Hunyuan3D2mvGenerator] Loaded mesh: %d vertices, %d faces"
              % (len(mesh.vertices), len(mesh.faces)))

        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

        self._report(progress_cb, 15, "Preprocessing front image...")
        front_image = self._preprocess(image_bytes, remove_bg=remove_bg)
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

        images = [front_image]
        for pct, view_name in ((22, "left"), (26, "back"), (30, "right")):
            image = self._optional_view_image(params, view_name, remove_bg)
            if image is not None:
                self._report(progress_cb, pct, "Preprocessing %s image..." % view_name)
                images.append(image)
                if cancel_event and cancel_event.is_set():
                    raise GenerationCancelled()

        print("[Hunyuan3D2mvGenerator] Using %d images for texturing" % len(images))

        self._report(progress_cb, 38, "Loading texture pipeline...")
        if self._texture_pipeline is None:
            self._load_texture()
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

        self._report(progress_cb, 50, "Generating textures...")
        stop_evt = threading.Event()
        progress_thread = None
        if progress_cb:
            progress_thread = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 50, 88, "Generating textures...", stop_evt),
                daemon=True,
            )
            progress_thread.start()

        try:
            with torch.no_grad():
                textured_mesh = self._texture_pipeline(mesh, image=images)
            print("[Hunyuan3D2mvGenerator] Texturing complete")
        except Exception as e:
            print("[Hunyuan3D2mvGenerator] Texturing failed: %s" % e)
            raise
        finally:
            stop_evt.set()
            if progress_thread:
                progress_thread.join(timeout=1.0)

        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

        self._report(progress_cb, 90, "Exporting textured GLB...")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.outputs_dir / (
            "%d_%s_textured.glb" % (int(time.time()), uuid.uuid4().hex[:8])
        )
        textured_mesh.export(str(out_path))
        print("[Hunyuan3D2mvGenerator] Exported textured GLB to: %s" % out_path)

        self._report(progress_cb, 100, "Done")
        return out_path
